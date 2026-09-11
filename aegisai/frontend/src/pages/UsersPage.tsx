/** Users page */
import { useEffect, useState } from 'react';
import {
  Search,
  Edit,
  Trash2,
  UserPlus,
  Loader2,
  User as UserIcon,
  Crown,
} from 'lucide-react';
import { api } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Card, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';
import { Dropdown } from '../components/ui/Dropdown';
import { getPageItems } from '../types';
import type { User, Role, Department } from '../types';

export const UsersPage = () => {
  const { user: currentUser } = useAuthStore();
  const [users, setUsers] = useState<User[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [departments, setDepartments] = useState<Department[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [deptFilter, setDeptFilter] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    full_name: '',
    role_id: '',
    department_id: '',
    is_active: true,
  });

  const fetchData = async () => {
    setLoading(true);
    try {
      const [usersRes, rolesRes, deptsRes] = await Promise.all([
        api.getUsers({ page, page_size: pageSize, search: search || undefined, role_id: roleFilter || undefined, department_id: deptFilter || undefined }),
        api.getRoles(),
        api.getDepartments(),
      ]);
      setUsers(getPageItems(usersRes));
      setTotal(usersRes.total);
      setRoles(rolesRes);
      setDepartments(deptsRes);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [page, pageSize, search, roleFilter, deptFilter]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      if (selectedUser) {
        await api.updateUser(selectedUser.id, formData);
      } else {
        await api.createUser({ ...formData, password: formData.password || 'TempPass123!' });
      }
      setShowCreateModal(false);
      setShowEditModal(false);
      fetchData();
    } catch (error) {
      console.error('Failed to save user:', error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (userId: string) => {
    if (!confirm('Are you sure you want to delete this user?')) return;
    try {
      await api.deleteUser(userId);
      fetchData();
    } catch (error) {
      console.error('Failed to delete user:', error);
    }
  };

  const openCreateModal = () => {
    setSelectedUser(null);
    setFormData({
      username: '',
      email: '',
      password: '',
      full_name: '',
      role_id: roles[0]?.id || '',
      department_id: '',
      is_active: true,
    });
    setShowCreateModal(true);
  };

  const openEditModal = (user: User) => {
    setSelectedUser(user);
    setFormData({
      username: user.username,
      email: user.email,
      password: '',
      full_name: user.full_name || '',
      role_id: user.role.id,
      department_id: user.department?.id || '',
      is_active: user.is_active,
    });
    setShowEditModal(true);
  };

  const roleBadges: Record<string, 'primary' | 'success' | 'warning' | 'danger' | 'gray'> = {
    admin: 'danger',
    manager: 'warning',
    engineer: 'primary',
    employee: 'gray',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-900">Users</h1>
          <p className="text-dark-500">Manage user accounts and permissions</p>
        </div>
        {currentUser?.role.name === 'admin' && (
          <Button onClick={openCreateModal}>
            <UserPlus className="h-4 w-4" />
            Add User
          </Button>
        )}
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="relative flex-1 min-w-[200px] max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dark-400" />
            <input
              type="search"
              placeholder="Search users..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg border border-dark-200 bg-white px-10 py-2 text-sm text-dark-900 placeholder:text-dark-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
            />
          </div>
          <Select
            options={[{ value: '', label: 'All Roles' }, ...roles.map(r => ({ value: r.id, label: r.name }))]}
            value={roleFilter}
            onChange={setRoleFilter}
            placeholder="Role"
            className="w-48"
          />
          <Select
            options={[{ value: '', label: 'All Departments' }, ...departments.map(d => ({ value: d.id, label: d.name }))]}
            value={deptFilter}
            onChange={setDeptFilter}
            placeholder="Department"
            className="w-48"
          />
        </div>
      </Card>

      {/* Users Table */}
      <Card>
        <CardContent className="p-0">
          {loading ? (
            <div className="p-8 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary-600 mx-auto" />
              <p className="mt-2 text-dark-500">Loading users...</p>
            </div>
          ) : users.length === 0 ? (
            <div className="p-8 text-center">
              <UserIcon className="h-12 w-12 text-dark-200 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-dark-700">No users found</h3>
              <p className="text-dark-500 mt-1">Add your first user to get started</p>
              {currentUser?.role.name === 'admin' && (
                <Button className="mt-4" onClick={openCreateModal}>
                  <UserPlus className="h-4 w-4" />
                  Add User
                </Button>
              )}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-200 bg-dark-50">
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider">User</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden md:table-cell">Role</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden lg:table-cell">Department</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden xl:table-cell">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden xl:table-cell">Last Login</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-dark-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-100">
                  {users.map((user) => (
                    <tr key={user.id} className="hover:bg-dark-50 transition-colors">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary-100">
                            {user.is_superuser ? (
                              <Crown className="h-5 w-5 text-primary-600" />
                            ) : (
                              <UserIcon className="h-5 w-5 text-primary-600" />
                            )}
                          </div>
                          <div>
                            <p className="font-medium text-dark-900">{user.username}</p>
                            <p className="text-xs text-dark-500">{user.email}</p>
                            {user.full_name && <p className="text-xs text-dark-400">{user.full_name}</p>}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 hidden md:table-cell">
                        <Badge variant={roleBadges[user?.role?.name ?? ''] || 'gray'}>
                          {(user?.role?.name ?? 'User').charAt(0).toUpperCase() + (user?.role?.name ?? 'user').slice(1)}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 hidden lg:table-cell text-sm text-dark-500">
                        {user.department?.name || '—'}
                      </td>
                      <td className="px-6 py-4 hidden xl:table-cell">
                        <Badge variant={user.is_active ? 'success' : 'gray'}>
                          {user.is_active ? 'Active' : 'Inactive'}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 hidden xl:table-cell text-sm text-dark-500">
                        {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                      </td>
                      <td className="px-6 py-4 text-right">
                        {currentUser?.role.name === 'admin' && user.id !== currentUser.id && (
                          <Dropdown
                            options={[
                              { value: 'edit', label: 'Edit', icon: <Edit className="h-4 w-4" /> },
                              { value: 'delete', label: 'Delete', icon: <Trash2 className="h-4 w-4" /> },
                            ]}
                            placeholder="Actions"
                            onChange={(action) => {
                              if (action === 'edit') openEditModal(user);
                              if (action === 'delete') handleDelete(user.id);
                            }}
                          />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {total > pageSize && (
            <div className="px-6 py-4 border-t border-dark-200 flex items-center justify-between">
              <p className="text-sm text-dark-500">
                Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, total)} of {total}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage(page - 1)}
                  disabled={page === 1}
                >
                  Previous
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage(page + 1)}
                  disabled={page * pageSize >= total}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Create/Edit User Modal */}
      <Modal
        isOpen={showCreateModal || showEditModal}
        onClose={() => { setShowCreateModal(false); setShowEditModal(false); }}
        title={selectedUser ? 'Edit User' : 'Create User'}
        size="lg"
      >
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Input
              name="username"
              label="Username"
              placeholder="john.doe"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              required
              disabled={!!selectedUser}
            />
            <Input
              name="email"
              label="Email"
              placeholder="john@company.com"
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              required
            />
          </div>

          <Input
            name="full_name"
            label="Full Name (optional)"
            placeholder="John Doe"
            value={formData.full_name}
            onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
          />

          <div className="grid gap-4 sm:grid-cols-2">
            <Select
              name="role_id"
              label="Role"
              options={roles.map(r => ({ value: r.id, label: r.name.charAt(0).toUpperCase() + r.name.slice(1) }))}
              value={formData.role_id}
              onChange={(value) => setFormData({ ...formData, role_id: value })}
              required
            />
            <Select
              name="department_id"
              label="Department (optional)"
              options={[{ value: '', label: 'None' }, ...departments.map(d => ({ value: d.id, label: d.name }))]}
              value={formData.department_id}
              onChange={(value) => setFormData({ ...formData, department_id: value })}
              placeholder="Select department"
            />
          </div>

          {!selectedUser && (
            <Input
              name="password"
              label="Password"
              type="password"
              placeholder="Min 8 characters"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              required
            />
          )}

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="is_active"
              checked={formData.is_active}
              onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
              className="rounded border-dark-300 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="is_active" className="text-sm text-dark-700">Active</label>
          </div>

          <div className="flex justify-end gap-2 pt-4">
            <Button type="button" variant="secondary" onClick={() => { setShowCreateModal(false); setShowEditModal(false); }}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={submitting}>
              {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
              {selectedUser ? 'Update' : 'Create'}
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
};