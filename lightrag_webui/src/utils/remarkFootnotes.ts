import { visit } from 'unist-util-visit'
import type { Plugin } from 'unified'
import type { Root, Text, Paragraph, Html } from 'mdast'

// Simple footnote plugin for remark
export const remarkFootnotes: Plugin<[], Root> = () => {
  return (tree: Root) => {
    const footnoteDefinitions = new Map<string, string>()

    // First pass: collect footnote definitions and remove them
    const nodesToRemove: Array<{ parent: any; index: number }> = []

    visit(tree, 'paragraph', (node: Paragraph, index, parent) => {
      if (!parent || typeof index !== 'number') return

      // Check if this paragraph contains only a footnote definition
      if (node.children.length === 1 && node.children[0].type === 'text') {
        const text = (node.children[0] as Text).value
        const match = text.match(/^\[\^([^\]]+)\]:\s*(.+)$/)
        if (match) {
          const [, id, content] = match
          footnoteDefinitions.set(id, content.trim())
          nodesToRemove.push({ parent, index })
          return
        }
      }
    })

    // Remove footnote definition paragraphs
    nodesToRemove.reverse().forEach(({ parent, index }) => {
      parent.children.splice(index, 1)
    })

    // Second pass: find footnote references and replace them
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

        // Add footnote reference as HTML
        replacements.push({
          type: 'html',
          value: `<sup><a href="#fn-${id}" id="fnref-${id}" class="footnote-ref">${id}</a></sup>`
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

    // Third pass: add footnotes section at the end if we have definitions
    if (footnoteDefinitions.size > 0) {
      const footnotesList: any[] = []

      footnoteDefinitions.forEach((content, id) => {
        footnotesList.push({
          type: 'listItem',
          children: [{
            type: 'paragraph',
            children: [{
              type: 'html',
              value: `<span id="fn-${id}">${content} <a href="#fnref-${id}" class="footnote-backref">↩</a></span>`
            }]
          }]
        })
      })

      // Add footnotes section
      tree.children.push({
        type: 'html',
        value: '<div class="footnotes">'
      } as Html)

      tree.children.push({
        type: 'list',
        ordered: true,
        start: 1,
        spread: false,
        children: footnotesList
      })

      tree.children.push({
        type: 'html',
        value: '</div>'
      } as Html)
    }
  }
}
