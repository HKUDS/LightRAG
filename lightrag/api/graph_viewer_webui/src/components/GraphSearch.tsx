import { FC, useCallback } from 'react'
import {
  EdgeById,
  NodeById,
  useGraphSearch,
  GraphSearchInputProps,
  GraphSearchContextProvider,
  GraphSearchContextProviderProps
} from '@react-sigma/graph-search'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { searchResultLimit } from '@/lib/constants'

interface OptionItem {
  id: string
  type: 'nodes' | 'edges' | 'message'
  message?: string
}

function OptionComponent(item: OptionItem) {
  return (
    <div>
      {item.type === 'nodes' && <NodeById id={item.id} />}
      {item.type === 'edges' && <EdgeById id={item.id} />}
      {item.type === 'message' && <div>{item.message}</div>}
    </div>
  )
}

const messageId = '__message_item'

/**
 * Component thats display the search input.
 */
export const GraphSearchInput = ({
  onChange,
  onFocus,
  type,
  value
}: {
  onChange: GraphSearchInputProps['onChange']
  onFocus?: GraphSearchInputProps['onFocus']
  type?: GraphSearchInputProps['type']
  value?: GraphSearchInputProps['value']
}) => {
  const { search } = useGraphSearch()

  /**
   * Loading the options while the user is typing.
   */
  const loadOptions = useCallback(
    async (query?: string): Promise<OptionItem[]> => {
      if (onFocus) onFocus(null)
      if (!query) return []
      const result = (await search(query, type)) as OptionItem[]

      // prettier-ignore
      return result.length <= searchResultLimit
        ? result
        : [
          ...result.slice(0, searchResultLimit),
          {
            type: 'message',
            id: messageId,
            message: `And ${result.length - searchResultLimit} others`
          }
        ]
    },
    [type, search, onFocus]
  )

  return (
    <AsyncSelect
      className="bg-background/60 w-52 rounded-xl border-1 opacity-60 backdrop-blur-lg transition-opacity hover:opacity-100"
      fetcher={loadOptions}
      renderOption={OptionComponent}
      getOptionValue={(item) => item.id}
      value={value && value.type !== 'message' ? value.id : null}
      onChange={(id) => {
        if (id !== messageId && type) onChange(id ? { id, type } : null)
      }}
      onFocus={(id) => {
        if (id !== messageId && onFocus && type) onFocus(id ? { id, type } : null)
      }}
      label={'item'}
      preload={false}
      placeholder="Type search here..."
    />
  )
}

/**
 * Component that display the search.
 */
const GraphSearch: FC<GraphSearchInputProps & GraphSearchContextProviderProps> = ({
  minisearchOptions,
  ...props
}) => (
  <GraphSearchContextProvider minisearchOptions={minisearchOptions}>
    <GraphSearchInput {...props} />
  </GraphSearchContextProvider>
)

export default GraphSearch
