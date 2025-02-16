import { backendBaseUrl } from '@/lib/constants'

export default function ApiSite() {
  return <iframe src={backendBaseUrl + '/docs'} className="size-full" />
}
