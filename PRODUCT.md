# Product

## Register

product

## Users

The primary user is the knowledge base maintainer working on a local LightRAG medical knowledge base. The maintainer needs to inspect graph structure, understand entity and relation quality, review source grounding, approve or reject proposed changes, and compare iteration results without being forced to read raw GraphML, JSON, or scattered Markdown files.

Secondary users may include medical domain reviewers and technical collaborators, but the first interface is optimized for one maintainer making careful, source-grounded decisions.

## Product Purpose

LightRAG WebUI should become a KG maintenance console for medical knowledge bases. It must make the graph understandable to a human, expose evidence and quality issues clearly, and guide iterative improvement through reviewable proposals rather than hidden automatic mutation.

Success means the maintainer can answer three questions quickly: what is in the current knowledge base, what is structurally or medically questionable, and which approved change should be made next.

## Brand Personality

Modern, layered, and trustworthy.

The interface should feel like a serious AI workbench for knowledge maintenance: precise enough for medical source review, visually rich enough to make graph structure legible, and restrained enough that users trust the decisions it asks them to make.

## Anti-references

Avoid consumer chatbot styling, marketing dashboards, decorative science-fiction panels, playful medical app tropes, overloaded node-link canvases without hierarchy, and any UI that makes generated analysis look like verified medical evidence.

Avoid presenting approval actions as casual one-click fixes. Medical fact changes, prompt changes, ontology/rule changes, hierarchy changes, relation changes, Web display changes, and workspace rebuilds must remain visibly gated.

## Design Principles

1. Evidence before confidence: every medical claim, node, relation, and proposed change should be traceable to source material or explicitly marked as a quality concern.
2. Graphs need hierarchy: the visual graph should organize disease, categories, subcategories, and concrete facts in a way humans can scan before they zoom into details.
3. Review is a workflow, not a modal: quality findings, proposals, approval decisions, and diff results should live in persistent work surfaces with clear status.
4. Keep generated analysis humble: LLM or agent output is useful as review material, but the UI must not imply that generated suggestions are source truth.
5. Dense but calm: the console should support repeated maintenance work with compact information, strong labels, predictable navigation, and restrained visual emphasis.

## Accessibility & Inclusion

Target WCAG AA contrast for text and controls. The graph view must not rely on color alone: node type, relation type, severity, and approval state need labels, icons, shapes, or textual legends. Motion should be brief and state-driven, with reduced-motion alternatives. Chinese and English text should remain supported through the existing i18n system.
