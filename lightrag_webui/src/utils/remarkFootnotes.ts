import type { Html, Parent, Root, RootContent, Text } from 'mdast'
import type { Plugin } from 'unified'
import { visit } from 'unist-util-visit'

// Simple footnote plugin for remark - only renders inline citations
export const remarkFootnotes: Plugin<[], Root> = () => {
  return (tree: Root) => {
    // Find footnote references and replace them with inline citations
    visit(tree, 'text', (node: Text, index, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number') return

      const text = node.value
      const footnoteRegex = /\[\^([^\]]+)\]/g
      let match: RegExpExecArray | null = footnoteRegex.exec(text)
      const replacements: RootContent[] = []
      let lastIndex = 0

      while (match !== null) {
        const [fullMatch, id] = match
        const startIndex = match.index

        // Add text before footnote
        if (startIndex > lastIndex) {
          const textNode: Text = {
            type: 'text',
            value: text.slice(lastIndex, startIndex),
          }
          replacements.push(textNode)
        }

        // Check if there's another footnote immediately following this one
        const nextIndex = startIndex + fullMatch.length
        const remainingText = text.slice(nextIndex)
        const hasConsecutiveFootnote = /^\[\^[^\]]+\]/.test(remainingText)

        // Add footnote reference as HTML with placeholder link
        const footnoteHtml = `<sup><a href="#footnote-${id}" class="footnote-ref">${id}</a></sup>`

        // Add spacing if there's a consecutive footnote
        const htmlWithSpacing = hasConsecutiveFootnote ? `${footnoteHtml}&nbsp;` : footnoteHtml

        const htmlNode: Html = {
          type: 'html',
          value: htmlWithSpacing,
        }
        replacements.push(htmlNode)

        lastIndex = startIndex + fullMatch.length
        match = footnoteRegex.exec(text)
      }

      // Add remaining text
      if (lastIndex < text.length) {
        const textNode: Text = {
          type: 'text',
          value: text.slice(lastIndex),
        }
        replacements.push(textNode)
      }

      // Replace the text node if we found footnotes
      if (replacements.length > 1) {
        parent.children.splice(index, 1, ...replacements)
      }
    })
  }
}
