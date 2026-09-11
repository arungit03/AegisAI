/** Main application layout with sidebar */
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { useState } from 'react';
import {
  LayoutDashboard,
  MessageSquare,
  FileText,
  Users,
  Settings,
  LogOut,
  Menu,
  Bell,
  Search,
  User,
  Shield,
  Activity,
} from 'lucide-react';
import { clsx } from 'clsx';
import { useAuthStore } from '../stores/authStore';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'Documents', href: '/documents', icon: FileText },
  { name: 'Users', href: '/users', icon: Users, roles: ['admin'] },
  { name: 'Audit Logs', href: '/audit-logs', icon: Shield, roles: ['admin'] },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export const MainLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const filteredNavigation = navigation.filter((item) => {
    if (!item.roles) return true;
    return user?.role && item.roles.includes(user.role.name);
  });

  return (
    <div className="flex min-h-screen bg-dark-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed top-0 left-0 z-50 h-screen bg-white border-r border-dark-200 transition-all duration-300 ease-in-out lg:relative lg:z-auto',
          sidebarCollapsed ? 'w-20' : 'w-64',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        )}
        aria-label="Main navigation"
      >
        {/* Logo */}
        <div className={clsx('flex items-center h-16 px-4 border-b border-dark-100', sidebarCollapsed && 'justify-center')}>
          {!sidebarCollapsed && (
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
                <Shield className="h-5 w-5 text-white" />
              </div>
              <span className="text-lg font-semibold text-dark-900">AegisAI</span>
            </div>
          )}
          {sidebarCollapsed && (
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
              <Shield className="h-5 w-5 text-white" />
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1" aria-label="Main">
          {filteredNavigation.map((item) => {
            const isActive = location.pathname === item.href || location.pathname.startsWith(item.href + '/');
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={clsx(
                  'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors duration-200',
                  isActive
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-dark-600 hover:bg-dark-100 hover:text-dark-900',
                  sidebarCollapsed && 'justify-center'
                )}
                title={sidebarCollapsed ? item.name : undefined}
                onClick={() => setSidebarOpen(false)}
                aria-current={isActive ? 'page' : undefined}
              >
                <item.icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
                {!sidebarCollapsed && <span>{item.name}</span>}
              </NavLink>
            );
          })}
        </nav>

        {/* Bottom section */}
        <div className="border-t border-dark-100 p-3">
          {!sidebarCollapsed && user && (
            <div className="flex items-center gap-3 px-3 py-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-100">
                <User className="h-5 w-5 text-primary-600" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-dark-900 truncate">{user.full_name || user.username}</p>
                <p className="text-xs text-dark-500 capitalize">{user.role.name}</p>
              </div>
            </div>
          )}

          <NavLink
            to="/settings"
            className={clsx(
              'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-dark-600 hover:bg-dark-100 hover:text-dark-900 transition-colors',
              sidebarCollapsed && 'justify-center'
            )}
            title={sidebarCollapsed ? 'Settings' : undefined}
            onClick={() => setSidebarOpen(false)}
          >
            <Settings className="h-5 w-5 flex-shrink-0" />
            {!sidebarCollapsed && <span>Settings</span>}
          </NavLink>

          <button
            onClick={handleLogout}
            className={clsx(
              'flex items-center gap-3 w-full rounded-lg px-3 py-2.5 text-sm font-medium text-dark-600 hover:bg-dark-100 hover:text-dark-900 transition-colors',
              sidebarCollapsed && 'justify-center mx-auto'
            )}
            title={sidebarCollapsed ? 'Logout' : undefined}
          >
            <LogOut className="h-5 w-5 flex-shrink-0" />
            {!sidebarCollapsed && <span>Logout</span>}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex min-w-0 flex-1 flex-col transition-all duration-300">
        {/* Top bar */}
        <header className="sticky top-0 z-30 h-16 bg-white/80 backdrop-blur-sm border-b border-dark-200">
          <div className="flex h-full items-center justify-between px-4 lg:px-6">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="lg:hidden p-2 rounded-lg text-dark-600 hover:bg-dark-100"
                aria-label="Toggle sidebar"
                aria-expanded={sidebarOpen}
              >
                <Menu className="h-6 w-6" />
              </button>

              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-lg text-dark-600 hover:bg-dark-100"
                aria-label="Open sidebar"
              >
                <Menu className="h-6 w-6" />
              </button>

              {!sidebarCollapsed && (
                <div className="relative w-full max-w-md hidden sm:block">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dark-400" />
                  <input
                    type="search"
                    placeholder="Search conversations, documents..."
                    className="w-full rounded-lg border border-dark-200 bg-dark-50 px-10 py-2 text-sm text-dark-900 placeholder:text-dark-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
                    aria-label="Search"
                  />
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button className="p-2 rounded-lg text-dark-600 hover:bg-dark-100 hover:text-dark-900 transition-colors" aria-label="Notifications">
                <Bell className="h-5 w-5" />
              </button>

              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-dark-50">
                <Activity className="h-3.5 w-3.5 text-green-500" />
                <span className="text-xs font-medium text-dark-700">Online</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-6" id="main-content" role="main">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
