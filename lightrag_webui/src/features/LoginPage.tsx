import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/state'
import { loginToServer } from '@/api/lightrag'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'

import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { ZapIcon } from 'lucide-react'
import ThemeToggle from '@/components/ThemeToggle'
import LanguageToggle from '@/components/LanguageToggle'

const LoginPage = () => {
  const navigate = useNavigate()
  const { login } = useAuthStore()
  const { t } = useTranslation()
  const [loading, setLoading] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!username || !password) {
      toast.error(t('login.errorEmptyFields'))
      return
    }

    try {
      setLoading(true)
      const response = await loginToServer(username, password)
      login(response.access_token)
      navigate('/')
      toast.success(t('login.successMessage'))
    } catch (error) {
      console.error('Login failed...', error)
      toast.error(t('login.errorInvalidCredentials'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-gray-900 dark:to-gray-800">
      <div className="absolute top-4 right-4 flex items-center gap-2">
        <LanguageToggle />
        <ThemeToggle />
      </div>
      <Card className="w-full max-w-[480px] shadow-lg mx-4">
        <CardHeader className="flex items-center justify-center space-y-2 pb-8 pt-6">
          <div className="flex flex-col items-center space-y-4">
            <div className="flex items-center gap-3">
              <img src="/logo.png" alt="LightRAG Logo" className="h-12 w-12" />
              <ZapIcon className="size-10 text-emerald-400" aria-hidden="true" />
            </div>
            <div className="text-center space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">LightRAG</h1>
              <p className="text-muted-foreground text-sm">
                {t('login.description')}
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="px-8 pb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="flex items-center gap-4">
              <label htmlFor="username" className="text-sm font-medium w-16 shrink-0">
                {t('login.username')}
              </label>
              <Input
                id="username"
                placeholder={t('login.usernamePlaceholder')}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="h-11 flex-1"
              />
            </div>
            <div className="flex items-center gap-4">
              <label htmlFor="password" className="text-sm font-medium w-16 shrink-0">
                {t('login.password')}
              </label>
              <Input
                id="password"
                type="password"
                placeholder={t('login.passwordPlaceholder')}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="h-11 flex-1"
              />
            </div>
            <Button
              type="submit"
              className="w-full h-11 text-base font-medium mt-2"
              disabled={loading}
            >
              {loading ? t('login.loggingIn') : t('login.loginButton')}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

export default LoginPage
