import { useState, useEffect, useCallback, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  BadgeCheck,
  KeyRound,
  RefreshCw,
  ShieldCheck,
  User as UserIcon,
  UserCog,
  UserPlus,
  Lock,
  Unlock,
  Menu,
  Shield,
  Pencil,
  Trash2,
  AlertCircle,
  CheckCircle2,
  XCircle
} from 'lucide-react'
import { toast } from 'sonner'

import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import Input from '@/components/ui/Input'
import Badge from '@/components/ui/Badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter
} from '@/components/ui/Dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import Checkbox from '@/components/ui/Checkbox'
import { useAuthStore } from '@/stores/state'
import {
  getUsers,
  createUser,
  updateUser,
  deleteUser,
  toggleLockUser,
  updateUserPermissions,
  AVAILABLE_MENU_ITEMS,
  type UserInfo
} from '@/api/lightrag'

const MENU_LABEL_MAP: Record<string, string> = {
  dashboard: 'header.dashboard',
  'knowledge-base': 'header.knowledgeBase',
  documents: 'header.documents',
  'knowledge-graph': 'header.knowledgeGraph',
  retrieval: 'header.retrieval',
  users: 'header.users'
}

const MENU_FALLBACK_MAP: Record<string, string> = {
  dashboard: 'Dashboard',
  'knowledge-base': 'Knowledge Base',
  documents: 'Document Management',
  'knowledge-graph': 'Knowledge Graph',
  retrieval: 'Retrieval',
  users: 'User Management'
}

function formatExpiry(expiresAt: number | null): string | null {
  if (!expiresAt) return null
  const d = new Date(expiresAt)
  if (isNaN(d.getTime())) return null
  return d.toLocaleString()
}

export default function UserManagement() {
  const { t } = useTranslation()
  const { username, isGuestMode, coreVersion, apiVersion, lastTokenRenewal, tokenExpiresAt } =
    useAuthStore()

  const [users, setUsers] = useState<UserInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [showPermDialog, setShowPermDialog] = useState(false)
  const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null)

  // Form state
  const [newUsername, setNewUsername] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newRole, setNewRole] = useState('user')
  const [newPermissions, setNewPermissions] = useState<string[]>([...AVAILABLE_MENU_ITEMS])

  const [editPassword, setEditPassword] = useState('')
  const [editRole, setEditRole] = useState('user')

  const [deleteConfirmText, setDeleteConfirmText] = useState('')

  const [saving, setSaving] = useState(false)

  // Change password state
  const [showChangePwdDialog, setShowChangePwdDialog] = useState(false)
  const [changePwdNew, setChangePwdNew] = useState('')
  const [changePwdConfirm, setChangePwdConfirm] = useState('')

  const role = useMemo(() => {
    try {
      const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
      if (!token) return null
      const payload = JSON.parse(atob(token.split('.')[1]))
      return typeof payload.role === 'string' ? payload.role : null
    } catch {
      return null
    }
  }, [])

  const fetchUsers = useCallback(async () => {
    setLoading(true)
    try {
      const result = await getUsers()
      setUsers(result.users)
    } catch {
      toast.error(t('userManagement.loadError', 'Failed to load users'))
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchUsers()
  }, [fetchUsers])

  const initials = useMemo(() => {
    if (!username) return '?'
    return username.slice(0, 1).toUpperCase()
  }, [username])

  const isAdmin = role === 'admin'

  // Add user
  const handleAddUser = async () => {
    if (!newUsername.trim()) {
      toast.error(t('userManagement.usernameRequired', 'Username is required'))
      return
    }
    if (!newPassword) {
      toast.error(t('userManagement.passwordRequired', 'Password is required'))
      return
    }
    setSaving(true)
    try {
      await createUser({
        username: newUsername.trim(),
        password: newPassword,
        role: newRole,
        permissions: newPermissions
      })
      toast.success(t('userManagement.userCreated', 'User created successfully'))
      setShowAddDialog(false)
      resetAddForm()
      await fetchUsers()
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error.message
      toast.error(t('userManagement.createFailed', 'Failed to create user') + `: ${msg}`)
    } finally {
      setSaving(false)
    }
  }

  // Change own password
  const handleChangePassword = async () => {
    if (!changePwdNew) {
      toast.error(t('userManagement.passwordRequired', 'Password is required'))
      return
    }
    if (changePwdNew !== changePwdConfirm) {
      toast.error(t('userManagement.passwordsDoNotMatch', 'Passwords do not match'))
      return
    }
    setSaving(true)
    try {
      await updateUser(username!, {
        password: changePwdNew
      })
      toast.success(t('userManagement.passwordChanged', 'Password changed successfully'))
      setShowChangePwdDialog(false)
      setChangePwdNew('')
      setChangePwdConfirm('')
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error.message
      toast.error(t('userManagement.passwordChangeFailed', 'Failed to change password') + `: ${msg}`)
    } finally {
      setSaving(false)
    }
  }

  const handleOpenChangePwd = () => {
    setChangePwdNew('')
    setChangePwdConfirm('')
    setShowChangePwdDialog(true)
  }

  const resetAddForm = () => {
    setNewUsername('')
    setNewPassword('')
    setNewRole('user')
    setNewPermissions([...AVAILABLE_MENU_ITEMS])
  }

  // Edit user
  const handleEditUser = async () => {
    if (!selectedUser) return
    setSaving(true)
    try {
      await updateUser(selectedUser.username, {
        password: editPassword || undefined,
        role: editRole
      })
      toast.success(t('userManagement.userUpdated', 'User updated successfully'))
      setShowEditDialog(false)
      await fetchUsers()
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error.message
      toast.error(t('userManagement.updateFailed', 'Failed to update user') + `: ${msg}`)
    } finally {
      setSaving(false)
    }
  }

  const openEditDialog = (user: UserInfo) => {
    setSelectedUser(user)
    setEditPassword('')
    setEditRole(user.role)
    setShowEditDialog(true)
  }

  // Delete user
  const handleDeleteUser = async () => {
    if (!selectedUser) return
    setSaving(true)
    try {
      await deleteUser(selectedUser.username)
      toast.success(t('userManagement.userDeleted', 'User deleted successfully'))
      setShowDeleteDialog(false)
      setDeleteConfirmText('')
      await fetchUsers()
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error.message
      toast.error(t('userManagement.deleteFailed', 'Failed to delete user') + `: ${msg}`)
    } finally {
      setSaving(false)
    }
  }

  const openDeleteDialog = (user: UserInfo) => {
    setSelectedUser(user)
    setDeleteConfirmText('')
    setShowDeleteDialog(true)
  }

  // Lock/unlock
  const handleToggleLock = async (user: UserInfo) => {
    try {
      await toggleLockUser(user.username, !user.locked)
      toast.success(
        user.locked
          ? t('userManagement.userUnlocked', 'User unlocked')
          : t('userManagement.userLocked', 'User locked')
      )
      await fetchUsers()
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error.message
      toast.error(t('userManagement.lockFailed', 'Failed to update lock status') + `: ${msg}`)
    }
  }

  // Permissions
  const handleOpenPermissions = (user: UserInfo) => {
    setSelectedUser(user)
    setNewPermissions([...user.permissions])
    setShowPermDialog(true)
  }

  const handleTogglePermission = (perm: string) => {
    setNewPermissions(prev =>
      prev.includes(perm)
        ? prev.filter(p => p !== perm)
        : [...prev, perm]
    )
  }

  const handleSavePermissions = async () => {
    if (!selectedUser) return
    setSaving(true)
    try {
      await updateUserPermissions(selectedUser.username, newPermissions)
      toast.success(t('userManagement.permissionsUpdated', 'Permissions updated'))
      setShowPermDialog(false)
      await fetchUsers()
    } catch (error: any) {
      const msg = error?.response?.data?.detail || error.message
      toast.error(t('userManagement.permissionsFailed', 'Failed to update permissions') + `: ${msg}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-[1200px] space-y-6 p-6">
      {/* Page header */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight">
            {isAdmin
              ? t('userManagement.title', 'User Management')
              : t('userManagement.personalInfo', 'Personal Information')}
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">
            {isAdmin
              ? t('userManagement.subtitle', 'Account, authentication and session information')
              : t('userManagement.personalInfoSubtitle', 'Account information and personal settings')}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isAdmin && (
            <Button onClick={() => { resetAddForm(); setShowAddDialog(true) }} className="gap-2">
              <UserPlus className="size-4" />
              {t('userManagement.addUser', 'Add User')}
            </Button>
          )}
        </div>
      </header>

      {/* Current account */}
      <Card variant="glass" className="glass-sheen overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <UserCog className="text-primary size-4" aria-hidden="true" />
            {t('userManagement.currentAccount', 'Current Account')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <span className="bg-primary/12 text-primary ring-primary/20 flex size-14 shrink-0 items-center justify-center rounded-2xl text-xl font-semibold ring-1 ring-inset">
              {initials}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-lg font-semibold">
                  {username || t('userManagement.unknownUser', 'Unknown user')}
                </p>
                {isGuestMode ? (
                  <Badge variant="secondary" className="gap-1">
                    <BadgeCheck className="size-3" />
                    {t('login.guestMode', 'Login Free')}
                  </Badge>
                ) : (
                  <Badge variant="secondary" className="gap-1 border-emerald-500/30 text-emerald-400">
                    <ShieldCheck className="size-3" />
                    {t('userManagement.standardUser', 'Standard User')}
                  </Badge>
                )}
                {isAdmin && (
                  <Badge variant="secondary" className="gap-1 border-amber-500/30 text-amber-400">
                    <Shield className="size-3" />
                    Admin
                  </Badge>
                )}
              </div>
              <p className="text-muted-foreground mt-0.5 text-sm">
                {t('userManagement.accountHint', 'Signed in to the RAG knowledge base server')}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Account details + system info */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card variant="glass" className="glass-sheen">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <KeyRound className="text-primary size-4" aria-hidden="true" />
              {t('userManagement.accountInfo', 'Account Information')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="divide-border/50 divide-y">
              <InfoRow
                label={t('login.username', 'Username')}
                value={username}
                mono
              />
              <InfoRow
                label={t('userManagement.authMode', 'Authentication Mode')}
                value={
                  isGuestMode
                    ? t('login.guestMode', 'Login Free')
                    : t('userManagement.authEnabled', 'Enabled')
                }
              />
              <InfoRow
                label={t('userManagement.role', 'Role')}
                value={role || t('userManagement.roleUnknown', 'unknown')}
                mono
              />
              <InfoRow
                label={t('userManagement.tokenExpires', 'Token Expires')}
                value={formatExpiry(tokenExpiresAt)}
              />
              <InfoRow
                label={t('userManagement.lastRenewal', 'Last Token Renewal')}
                value={lastTokenRenewal}
              />
            </div>
            {!isGuestMode && (
              <div className="mt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleOpenChangePwd}
                  className="gap-2"
                >
                  <KeyRound className="size-3.5" />
                  {t('userManagement.changePassword', 'Change Password')}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card variant="glass" className="glass-sheen">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <RefreshCw className="text-primary size-4" aria-hidden="true" />
              {t('userManagement.systemInfo', 'System Information')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="divide-border/50 divide-y">
              <InfoRow
                label={t('dashboard.version', 'Version')}
                value={
                  coreVersion && apiVersion
                    ? `${coreVersion} / ${apiVersion}`
                    : coreVersion || apiVersion
                }
                mono
              />
              <InfoRow
                label={t('userManagement.coreVersion', 'Core Version')}
                value={coreVersion}
                mono
              />
              <InfoRow
                label={t('userManagement.apiVersion', 'API Version')}
                value={apiVersion}
                mono
              />
            </div>
            <div className="text-muted-foreground mt-4 flex items-center gap-2 text-xs">
              <UserIcon className="size-3.5" aria-hidden="true" />
              {t('userManagement.sessionHint', 'Session data is stored locally in this browser')}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* User list (admin only) */}
      {isAdmin && (
        <Card variant="glass" className="glass-sheen overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <Shield className="text-primary size-4" aria-hidden="true" />
              {t('userManagement.userList', 'User List')}
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchUsers}
              disabled={loading}
            >
              <RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-border/50 border-b text-left">
                    <th className="px-4 py-3 font-medium text-xs text-muted-foreground">
                      {t('userManagement.colUsername', 'Username')}
                    </th>
                    <th className="px-4 py-3 font-medium text-xs text-muted-foreground">
                      {t('userManagement.colRole', 'Role')}
                    </th>
                    <th className="px-4 py-3 font-medium text-xs text-muted-foreground">
                      {t('userManagement.colStatus', 'Status')}
                    </th>
                    <th className="px-4 py-3 font-medium text-xs text-muted-foreground">
                      {t('userManagement.colPermissions', 'Menu Permissions')}
                    </th>
                    <th className="px-4 py-3 font-medium text-xs text-muted-foreground text-right">
                      {t('userManagement.colActions', 'Actions')}
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-border/50 divide-y">
                  {users.map(user => (
                    <tr key={user.username} className="hover:bg-muted/30 transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <UserIcon className="size-3.5 text-muted-foreground" />
                          <span className="font-medium">{user.username}</span>
                          {user.username === 'admin' && (
                            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                              Admin
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <Badge
                          variant="secondary"
                          className={
                            user.role === 'admin'
                              ? 'border-amber-500/30 text-amber-400'
                              : 'border-blue-500/30 text-blue-400'
                          }
                        >
                          {user.role}
                        </Badge>
                      </td>
                      <td className="px-4 py-3">
                        {user.locked ? (
                          <Badge variant="secondary" className="border-rose-500/30 text-rose-400 gap-1">
                            <XCircle className="size-3" />
                            {t('userManagement.locked', 'Locked')}
                          </Badge>
                        ) : (
                          <Badge variant="secondary" className="border-emerald-500/30 text-emerald-400 gap-1">
                            <CheckCircle2 className="size-3" />
                            {t('userManagement.active', 'Active')}
                          </Badge>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          {user.permissions.slice(0, 3).map(perm => (
                            <Badge key={perm} variant="outline" className="text-[10px] px-1.5 py-0">
                              {t(MENU_LABEL_MAP[perm] || perm, MENU_FALLBACK_MAP[perm] || perm)}
                            </Badge>
                          ))}
                          {user.permissions.length > 3 && (
                            <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                              +{user.permissions.length - 3}
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => openEditDialog(user)}
                            title={t('userManagement.editUser', 'Edit User')}
                          >
                            <Pencil className="size-3.5" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleToggleLock(user)}
                            title={
                              user.locked
                                ? t('userManagement.unlockUser', 'Unlock User')
                                : t('userManagement.lockUser', 'Lock User')
                            }
                            disabled={user.username === 'admin'}
                          >
                            {user.locked ? (
                              <Unlock className="size-3.5 text-emerald-400" />
                            ) : (
                              <Lock className="size-3.5 text-amber-400" />
                            )}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleOpenPermissions(user)}
                            title={t('userManagement.managePermissions', 'Manage Permissions')}
                          >
                            <Menu className="size-3.5" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => openDeleteDialog(user)}
                            title={t('userManagement.deleteUser', 'Delete User')}
                            disabled={user.username === 'admin'}
                            className="text-rose-400 hover:text-rose-300 hover:bg-rose-500/10"
                          >
                            <Trash2 className="size-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {users.length === 0 && !loading && (
                    <tr>
                      <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground text-sm">
                        {t('userManagement.noUsers', 'No users found')}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Add User Dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('userManagement.addUser', 'Add User')}</DialogTitle>
            <DialogDescription>
              {t('userManagement.addUserDesc', 'Create a new user account')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.username', 'Username')}
              </label>
              <Input
                placeholder={t('userManagement.usernamePlaceholder', 'Enter username')}
                value={newUsername}
                onChange={e => setNewUsername(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.password', 'Password')}
              </label>
              <Input
                type="password"
                placeholder={t('userManagement.passwordPlaceholder', 'Enter password')}
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.role', 'Role')}
              </label>
              <Select value={newRole} onValueChange={setNewRole}>
                <SelectTrigger>
                  <SelectValue placeholder={t('userManagement.selectRole', 'Select role')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">User</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.menuPermissions', 'Menu Permissions')}
              </label>
              <div className="grid grid-cols-2 gap-2 rounded-md border p-3">
                {AVAILABLE_MENU_ITEMS.map(perm => (
                  <label key={perm} className="flex items-center gap-2 text-sm cursor-pointer">
                    <Checkbox
                      checked={newPermissions.includes(perm)}
                      onCheckedChange={() => handleTogglePermission(perm)}
                    />
                    {t(MENU_LABEL_MAP[perm], MENU_FALLBACK_MAP[perm])}
                  </label>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowAddDialog(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={handleAddUser} disabled={saving}>
              {saving ? t('common.saving', 'Saving...') : t('userManagement.create', 'Create')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit User Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t('userManagement.editUser', 'Edit User')}: {selectedUser?.username}
            </DialogTitle>
            <DialogDescription>
              {t('userManagement.editUserDesc', 'Update user password or role')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.newPassword', 'New Password')}
              </label>
              <Input
                type="password"
                placeholder={t('userManagement.passwordLeaveEmpty', 'Leave empty to keep current')}
                value={editPassword}
                onChange={e => setEditPassword(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.role', 'Role')}
              </label>
              <Select value={editRole} onValueChange={setEditRole}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">User</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowEditDialog(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={handleEditUser} disabled={saving}>
              {saving ? t('common.saving', 'Saving...') : t('common.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete User Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-rose-400">
              <AlertCircle className="size-5" />
              {t('userManagement.deleteUser', 'Delete User')}
            </DialogTitle>
            <DialogDescription>
              {t('userManagement.deleteUserDesc', 'Are you sure you want to delete this user? This action cannot be undone.')}
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <p className="text-sm font-medium mb-4">
              {t('userManagement.deleteUserConfirm', 'Type the username to confirm')}: <strong>{selectedUser?.username}</strong>
            </p>
            <Input
              placeholder={t('userManagement.deleteUserPlaceholder', 'Type username to confirm')}
              value={deleteConfirmText}
              onChange={e => setDeleteConfirmText(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowDeleteDialog(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteUser}
              disabled={deleteConfirmText !== selectedUser?.username || saving}
            >
              {saving ? t('common.saving', 'Saving...') : t('userManagement.delete', 'Delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Permissions Dialog */}
      <Dialog open={showPermDialog} onOpenChange={setShowPermDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Menu className="size-5" />
              {t('userManagement.managePermissions', 'Manage Permissions')}: {selectedUser?.username}
            </DialogTitle>
            <DialogDescription>
              {t('userManagement.permissionsDesc', 'Select which menu items this user can access')}
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <div className="grid grid-cols-2 gap-3 rounded-md border p-4">
              {AVAILABLE_MENU_ITEMS.map(perm => (
                <label key={perm} className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox
                    checked={newPermissions.includes(perm)}
                    onCheckedChange={() => handleTogglePermission(perm)}
                  />
                  {t(MENU_LABEL_MAP[perm], MENU_FALLBACK_MAP[perm])}
                </label>
              ))}
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowPermDialog(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={handleSavePermissions} disabled={saving}>
              {saving ? t('common.saving', 'Saving...') : t('common.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Change Password Dialog */}
      <Dialog open={showChangePwdDialog} onOpenChange={setShowChangePwdDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="size-5" />
              {t('userManagement.changePassword', 'Change Password')}
            </DialogTitle>
            <DialogDescription>
              {t('userManagement.changePasswordDesc', 'Change the password for your account')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.newPassword', 'New Password')}
              </label>
              <Input
                type="password"
                placeholder={t('userManagement.passwordPlaceholder', 'Enter new password')}
                value={changePwdNew}
                onChange={e => setChangePwdNew(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {t('userManagement.confirmPassword', 'Confirm Password')}
              </label>
              <Input
                type="password"
                placeholder={t('userManagement.confirmPassword', 'Confirm new password')}
                value={changePwdConfirm}
                onChange={e => setChangePwdConfirm(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowChangePwdDialog(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={handleChangePassword} disabled={saving}>
              {saving ? t('common.saving', 'Saving...') : t('common.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

interface InfoRowProps {
  label: string
  value?: string | null
  mono?: boolean
}

function InfoRow({ label, value, mono }: InfoRowProps) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <span className="text-muted-foreground shrink-0 text-xs">{label}</span>
      <span
        className={`truncate text-right text-xs font-medium ${mono ? 'font-mono' : ''}`}
        title={String(value)}
      >
        {String(value)}
      </span>
    </div>
  )
}