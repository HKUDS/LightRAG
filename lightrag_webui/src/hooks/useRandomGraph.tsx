import { Faker, en, faker as fak } from '@faker-js/faker'
import Graph, { UndirectedGraph } from 'graphology'
import erdosRenyi from 'graphology-generators/random/erdos-renyi'
import { useCallback, useEffect, useState } from 'react'
import seedrandom from 'seedrandom'
import { randomColor } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'

export type NodeType = {
  x: number
  y: number
  label: string
  size: number
  color: string
  highlighted?: boolean
}
export type EdgeType = { label: string }

/**
 * The goal of this file is to seed random generators if the query params 'seed' is present.
 */
const useRandomGraph = () => {
  const [faker, setFaker] = useState<Faker>(fak)

  useEffect(() => {
    // Globally seed the Math.random
    const params = new URLSearchParams(document.location.search)
    const seed = params.get('seed') // is the string "Jonathan"
    if (seed) {
      seedrandom(seed, { global: true })
      // seed faker with the random function
      const f = new Faker({ locale: en })
      f.seed(Math.random())
      setFaker(f)
    }
  }, [])

  const randomGraph = useCallback(() => {
    useGraphStore.getState().reset()

    // Create the graph
    const graph = erdosRenyi(UndirectedGraph, { order: 100, probability: 0.1 })
    graph.nodes().forEach((node: string) => {
      graph.mergeNodeAttributes(node, {
        label: faker.person.fullName(),
        size: faker.number.int({ min: Constants.minNodeSize, max: Constants.maxNodeSize }),
        color: randomColor(),
        x: Math.random(),
        y: Math.random(),
        // for node-border
        borderColor: randomColor(),
        borderSize: faker.number.float({ min: 0, max: 1, multipleOf: 0.1 }),
        // for node-image
        pictoColor: randomColor(),
        image: faker.image.urlLoremFlickr()
      })
    })
    
    // Add edge attributes
    graph.edges().forEach((edge: string) => {
      graph.mergeEdgeAttributes(edge, {
        label: faker.lorem.words(faker.number.int({ min: 1, max: 3 })),
        size: faker.number.float({ min: 1, max: 5 }),
        color: randomColor()
      })
    })
    
    return graph as Graph<NodeType, EdgeType>
  }, [faker])

  return { faker, randomColor, randomGraph }
}

export default useRandomGraph
