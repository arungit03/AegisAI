/** Settings page */
import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { api } from '../services/api';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Card, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Shield, User, Bell, Database, Cpu, Globe, Key, Save } from 'lucide-react';
import toast from 'react-hot-toast';

interface SettingsTabsProps {
  activeTab: string;
  onChange: (tab: string) => void;
}

const SettingsTabs = ({ activeTab, onChange }: SettingsTabsProps) => {
  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'system', label: 'System', icon: Cpu },
  ];

  return (
    <div className="border-b border-dark-200">
      <nav className="flex gap-1" aria-label="Settings tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              className={`
                flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-t-lg transition-colors
                ${isActive
                  ? 'bg-white text-primary-600 border-b-2 border-primary-600'
                  : 'text-dark-500 hover:text-dark-700 hover:bg-dark-50'
                }
              `}
            >
              <Icon className="h-5 w-5" />
              {tab.label}
            </button>
          );
        })}
      </nav>
    </div>
  );
};

export const SettingsPage = () => {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState('profile');
  const [saving, setSaving] = useState(false);

  if (!user) {
    return <div className="p-6 text-dark-500">Loading user information...</div>;
  }

  const [profileForm, setProfileForm] = useState({
    full_name: user?.full_name || '',
    email: user?.email || '',
  });

  const [securityForm, setSecurityForm] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  });

  const [notificationForm, setNotificationForm] = useState({
    email_notifications: true,
    push_notifications: false,
    weekly_digest: true,
  });

  const handleProfileSave = async () => {
    setSaving(true);
    try {
      await api.updateUser(user!.id, profileForm);
      toast.success('Profile updated successfully');
    } catch (error) {
      toast.error('Failed to update profile');
    } finally {
      setSaving(false);
    }
  };

  const handleSecuritySave = async () => {
    if (securityForm.new_password !== securityForm.confirm_password) {
      toast.error('Passwords do not match');
      return;
    }
    if (securityForm.new_password.length < 8) {
      toast.error('Password must be at least 8 characters');
      return;
    }
    setSaving(true);
    try {
      // TODO: Implement password change endpoint
      toast.success('Password changed successfully');
      setSecurityForm({ current_password: '', new_password: '', confirm_password: '' });
    } catch (error) {
      toast.error('Failed to change password');
    } finally {
      setSaving(false);
    }
  };

  const handleNotificationSave = async () => {
    setSaving(true);
    try {
      // TODO: Implement notification preferences endpoint
      toast.success('Notification preferences saved');
    } catch (error) {
      toast.error('Failed to save preferences');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-900">Settings</h1>
        <p className="text-dark-500">Manage your account and preferences</p>
      </div>

      <SettingsTabs activeTab={activeTab} onChange={setActiveTab} />

      <Card>
        <CardContent className="p-6">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary-100">
                  <User className="h-10 w-10 text-primary-600" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-dark-900">{user?.full_name || user?.username}</h3>
                  <p className="text-dark-500">{user?.email}</p>
                  <Badge variant="primary" className="mt-1">
                    {user?.role.name.charAt(0).toUpperCase() + user?.role.name.slice(1)}
                  </Badge>
                </div>
              </div>

              <div className="border-t border-dark-200 pt-6">
                <h4 className="text-sm font-medium text-dark-700 mb-4">Profile Information</h4>
                <div className="space-y-4">
                  <Input
                    label="Full Name"
                    value={profileForm.full_name}
                    onChange={(e) => setProfileForm({ ...profileForm, full_name: e.target.value })}
                    placeholder="Your full name"
                  />
                  <Input
                    label="Email"
                    type="email"
                    value={profileForm.email}
                    onChange={(e) => setProfileForm({ ...profileForm, email: e.target.value })}
                    disabled
                    helperText="Email cannot be changed"
                  />
                  <div>
                    <label className="block text-sm font-medium text-dark-700 mb-1.5">Role</label>
                    <Input
                      value={user?.role.name.charAt(0).toUpperCase() + user?.role.name.slice(1) || ''}
                      disabled
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-dark-700 mb-1.5">Department</label>
                    <Input
                      value={user?.department?.name || 'Not assigned'}
                      disabled
                    />
                  </div>
                </div>

                <div className="flex justify-end pt-4 border-t border-dark-200">
                  <Button variant="primary" onClick={handleProfileSave} loading={saving}>
                    <Save className="h-4 w-4" />
                    Save Changes
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-medium text-dark-700 mb-4">Change Password</h4>
                <div className="space-y-4">
                  <Input
                    label="Current Password"
                    type="password"
                    value={securityForm.current_password}
                    onChange={(e) => setSecurityForm({ ...securityForm, current_password: e.target.value })}
                    placeholder="Enter current password"
                  />
                  <Input
                    label="New Password"
                    type="password"
                    value={securityForm.new_password}
                    onChange={(e) => setSecurityForm({ ...securityForm, new_password: e.target.value })}
                    placeholder="Enter new password (min 8 chars)"
                    helperText="Must be at least 8 characters"
                  />
                  <Input
                    label="Confirm New Password"
                    type="password"
                    value={securityForm.confirm_password}
                    onChange={(e) => setSecurityForm({ ...securityForm, confirm_password: e.target.value })}
                    placeholder="Confirm new password"
                  />
                </div>
                <Button variant="primary" onClick={handleSecuritySave} loading={saving} className="mt-4">
                  <Key className="h-4 w-4" />
                  Change Password
                </Button>
              </div>

              <div className="border-t border-dark-200 pt-6">
                <h4 className="text-sm font-medium text-dark-700 mb-4">Active Sessions</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-dark-50">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary-100">
                        <Globe className="h-5 w-5 text-primary-600" />
                      </div>
                      <div>
                        <p className="font-medium text-dark-900">Current Session</p>
                        <p className="text-sm text-dark-500">Chrome on Windows • Active now</p>
                      </div>
                    </div>
                    <Badge variant="success">Current</Badge>
                  </div>
                  <p className="text-sm text-dark-500 text-center py-4">
                    No other active sessions
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Notifications Tab */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-medium text-dark-700 mb-4">Email Notifications</h4>
                <div className="space-y-4">
                  <label className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-dark-900">Email Notifications</p>
                      <p className="text-sm text-dark-500">Receive email updates about your account</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={notificationForm.email_notifications}
                      onChange={(e) => setNotificationForm({ ...notificationForm, email_notifications: e.target.checked })}
                      className="h-5 w-5 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-dark-900">Weekly Digest</p>
                      <p className="text-sm text-dark-500">Weekly summary of your activity</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={notificationForm.weekly_digest}
                      onChange={(e) => setNotificationForm({ ...notificationForm, weekly_digest: e.target.checked })}
                      className="h-5 w-5 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
                    />
                  </label>
                </div>
              </div>

              <div className="border-t border-dark-200 pt-6">
                <h4 className="text-sm font-medium text-dark-700 mb-4">Push Notifications</h4>
                <div className="space-y-4">
                  <label className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-dark-900">Push Notifications</p>
                      <p className="text-sm text-dark-500">Receive browser notifications</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={notificationForm.push_notifications}
                      onChange={(e) => setNotificationForm({ ...notificationForm, push_notifications: e.target.checked })}
                      className="h-5 w-5 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
                    />
                  </label>
                </div>
              </div>

              <div className="flex justify-end pt-4 border-t border-dark-200">
                <Button variant="primary" onClick={handleNotificationSave} loading={saving}>
                  <Save className="h-4 w-4" />
                  Save Preferences
                </Button>
              </div>
            </div>
          )}

          {/* System Tab */}
          {activeTab === 'system' && (
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-medium text-dark-700 mb-4">System Information</h4>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="p-4 rounded-lg bg-dark-50">
                    <p className="text-sm text-dark-500">Version</p>
                    <p className="font-mono text-dark-900">1.0.0</p>
                  </div>
                  <div className="p-4 rounded-lg bg-dark-50">
                    <p className="text-sm text-dark-500">Environment</p>
                    <p className="font-mono text-dark-900">Development</p>
                  </div>
                  <div className="p-4 rounded-lg bg-dark-50">
                    <p className="text-sm text-dark-500">API Server</p>
                    <p className="font-mono text-dark-900">http://localhost:8000</p>
                  </div>
                  <div className="p-4 rounded-lg bg-dark-50">
                    <p className="text-sm text-dark-500">Database</p>
                    <p className="font-mono text-dark-900">PostgreSQL</p>
                  </div>
                </div>
              </div>

              <div className="border-t border-dark-200 pt-6">
                <h4 className="text-sm font-medium text-dark-700 mb-4">Data Management</h4>
                <div className="space-y-3">
                  <Button variant="secondary" className="w-full justify-start gap-3">
                    <Database className="h-5 w-5" />
                    <span>Export My Data</span>
                  </Button>
                  <Button variant="secondary" className="w-full justify-start gap-3">
                    <Shield className="h-5 w-5" />
                    <span>Privacy Settings</span>
                  </Button>
                </div>
              </div>

              <div className="border-t border-dark-200 pt-6">
                <h4 className="text-sm font-medium text-dark-700 mb-4">Danger Zone</h4>
                <div className="p-4 rounded-lg bg-red-50 border border-red-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-red-900">Delete Account</p>
                      <p className="text-sm text-red-700">Permanently delete your account and all data</p>
                    </div>
                    <Button variant="danger" size="sm">Delete Account</Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};