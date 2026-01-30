import { useState } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/state'
import { register } from '@/api/lightrag'
import { KeyRound, User, Users, ArrowRight } from 'lucide-react'

const RegisterPage = () => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [orgId, setOrgId] = useState('org_default')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()
  const { login } = useAuthStore()

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!username || !password || !confirmPassword) {
      toast.error('Please fill in all fields')
      return
    }

    if (password !== confirmPassword) {
      toast.error('Passwords do not match')
      return
    }

    setIsLoading(true)
    try {
      const response = await register({
        username,
        password,
        org_id: orgId
      })

      if (response.access_token) {
        // Auto login after successful registration
        login(
          response.access_token,
          false, // isGuest
          response.core_version,
          response.api_version,
          response.webui_title,
          response.webui_description
        )
        toast.success('Registration successful! Logging in...')
        navigate('/')
      }
    } catch (error: any) {
      console.error('Registration failed:', error)
      const detail = error.response?.data?.detail
      if (typeof detail === 'string') {
        toast.error(`Registration failed: ${detail}`)
      } else if (error.message) {
        toast.error(`Registration failed: ${error.message}`)
      } else {
        toast.error('Registration failed. Please try again.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md border-border/40 shadow-xl bg-card/60 backdrop-blur-sm">
        <CardHeader className="space-y-2 text-center">
          <div className="flex justify-center mb-4">
            <div className="p-3 rounded-full bg-primary/10">
              <Users className="w-8 h-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-2xl font-bold tracking-tight">Create an Account</CardTitle>
          <CardDescription>
            Enter your details to register for LightRAG
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleRegister}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="username">Username</label>
              <div className="relative">
                <User className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter username"
                  className="pl-9"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoFocus
                  disabled={isLoading}
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label htmlFor="password">Password</label>
              <div className="relative">
                <KeyRound className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter password"
                  className="pl-9"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label htmlFor="confirm-password">Confirm Password</label>
              <div className="relative">
                <KeyRound className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="confirm-password"
                  type="password"
                  placeholder="Confirm password"
                  className="pl-9"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={isLoading}
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <label htmlFor="org-id">Organization ID</label>
              <div className="relative">
                <Input
                  id="org-id"
                  type="text"
                  placeholder="default"
                  value={orgId}
                  onChange={(e) => setOrgId(e.target.value)}
                  disabled={isLoading}
                />
                <p className="text-[0.8rem] text-muted-foreground mt-1">Leave as &apos;default&apos; or enter your organization ID.</p>
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex flex-col gap-4">
            <Button
              type="submit"
              className="w-full"
              disabled={isLoading}
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  <span>Creating Account...</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span>Register</span>
                  <ArrowRight className="h-4 w-4" />
                </div>
              )}
            </Button>
            <div className="text-center text-sm">
              <span className="text-muted-foreground">Already have an account? </span>
              <Button
                variant="link"
                className="p-0 h-auto font-medium"
                onClick={() => navigate('/login')}
                type="button"
              >
                Sign in
              </Button>
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  )
}

export default RegisterPage
