import { visit } from 'unist-util-visit'
import type { Plugin } from 'unified'
import type { Root, Text } from 'mdast'

// Simple footnote plugin for remark - only renders inline citations
export const remarkFootnotes: Plugin<[], Root> = () => {
  return (tree: Root) => {
    // Find footnote references and replace them with inline citations
    visit(tree, 'text', (node: Text, index, parent) => {
      if (!parent || typeof index !== 'number') return

      const text = node.value
      const footnoteRegex = /\[\^([^\]]+)\]/g
      let match
      const replacements: any[] = []
      let lastIndex = 0

      while ((match = footnoteRegex.exec(text)) !== null) {
        const [fullMatch, id] = match
        const startIndex = match.index!

        // Add text before footnote
        if (startIndex > lastIndex) {
          replacements.push({
            type: 'text',
            value: text.slice(lastIndex, startIndex)
          })
        }

        // Add footnote reference as HTML with placeholder link
        replacements.push({
          type: 'html',
          value: `<sup><a href="#footnote" class="footnote-ref">${id}</a></sup>`
        })

        lastIndex = startIndex + fullMatch.length
      }

      // Add remaining text
      if (lastIndex < text.length) {
        replacements.push({
          type: 'text',
          value: text.slice(lastIndex)
        })
      }

      // Replace the text node if we found footnotes
      if (replacements.length > 1) {
        parent.children.splice(index, 1, ...replacements)
      }
    })
  }
}
