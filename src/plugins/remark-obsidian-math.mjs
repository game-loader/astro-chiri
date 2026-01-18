import { visit } from 'unist-util-visit'

function splitInlineMath(value) {
  const nodes = []
  let index = 0

  while (index < value.length) {
    const start = value.indexOf('\\(', index)
    if (start === -1) break

    const end = value.indexOf('\\)', start + 2)
    if (end === -1) break

    if (start > index) {
      nodes.push({ type: 'text', value: value.slice(index, start) })
    }

    const mathValue = value.slice(start + 2, end)
    nodes.push({ type: 'inlineMath', value: mathValue })
    index = end + 2
  }

  if (index < value.length) {
    nodes.push({ type: 'text', value: value.slice(index) })
  }

  return nodes
}

function paragraphText(node) {
  const parts = []

  for (const child of node.children || []) {
    if (child.type === 'text') {
      parts.push(child.value)
    } else if (child.type === 'break') {
      parts.push('\n')
    } else {
      return null
    }
  }

  return parts.join('')
}

export default function remarkObsidianMath() {
  return (tree) => {
    visit(tree, 'paragraph', (node, index, parent) => {
      if (!parent || typeof index !== 'number') return

      const text = paragraphText(node)
      if (text === null) return

      const trimmed = text.trim()
      const blockMatch =
        trimmed.match(/^\\\[([\s\S]+)\\\]$/) ||
        trimmed.match(/^\$\$([\s\S]+)\$\$$/)

      if (!blockMatch) return

      parent.children.splice(index, 1, { type: 'math', value: blockMatch[1].trim() })
      return index
    })

    visit(tree, 'text', (node, index, parent) => {
      if (!parent || typeof index !== 'number') return
      if (!node.value.includes('\\(')) return
      if (parent.type === 'inlineCode' || parent.type === 'code' || parent.type === 'math') return

      const nodes = splitInlineMath(node.value)
      if (nodes.length === 1 && nodes[0].type === 'text') return

      parent.children.splice(index, 1, ...nodes)
      return index + nodes.length
    })
  }
}
