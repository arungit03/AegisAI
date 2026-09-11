/** Audit logs page */
import { useEffect, useState } from 'react';
import {
  Search,
  Filter,
  RefreshCw,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Clock,
  Shield,
} from 'lucide-react';
import { format } from 'date-fns';
import { clsx } from 'clsx';
import { api } from '../services/api';
import { getPageItems } from '../types';
import type { AuditLog } from '../types';
import { Button } from '../components/ui/Button';
import { Select } from '../components/ui/Select';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

const actionOptions = [
  { value: '', label: 'All Actions' },
  { value: 'login', label: 'Login' },
  { value: 'logout', label: 'Logout' },
  { value: 'document_uploaded', label: 'Document Uploaded' },
  { value: 'document_updated', label: 'Document Updated' },
  { value: 'document_deleted', label: 'Document Deleted' },
  { value: 'chat_query', label: 'Chat Query' },
  { value: 'user_created', label: 'User Created' },
  { value: 'user_updated', label: 'User Updated' },
  { value: 'user_deleted', label: 'User Deleted' },
];

export const AuditLogsPage = () => {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [actionFilter, setActionFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState<string | ''>('');

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await api.getAuditLogs({
        page,
        page_size: pageSize,
        action: actionFilter || undefined,
        success: statusFilter === '' ? undefined : statusFilter === 'success',
      });
      setLogs(getPageItems(response));
      setTotal(response.total);
    } catch (error) {
      console.error('Failed to fetch audit logs:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, [page, pageSize, actionFilter, statusFilter]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchLogs();
  };

  const statusColors: Record<'success' | 'error', 'primary' | 'success' | 'warning' | 'danger' | 'gray' | 'outline'> = {
    success: 'success',
    error: 'danger',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-900">Audit Logs</h1>
          <p className="text-dark-500">View and search audit trail events</p>
        </div>
        <Button variant="secondary" onClick={fetchLogs} disabled={loading}>
          <RefreshCw className={clsx('h-4 w-4', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-4">
          <form onSubmit={handleSearch} className="relative flex-1 min-w-[200px] max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dark-400" />
            <input
              type="search"
              placeholder="Search logs..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg border border-dark-200 bg-white px-10 py-2 text-sm text-dark-900 placeholder:text-dark-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
            />
          </form>
          <Select
            options={actionOptions}
            value={actionFilter}
            onChange={(value) => { setActionFilter(value); setPage(1); }}
            placeholder="Action"
            className="w-48"
          />
          <Select
            options={[
              { value: '', label: 'All Status' },
              { value: 'success', label: 'Success' },
              { value: 'error', label: 'Error' },
            ]}
            value={statusFilter}
            onChange={(value) => { setStatusFilter(value); setPage(1); }}
            placeholder="Status"
            className="w-36"
          />
          <div className="flex items-center gap-2 text-sm text-dark-500">
            <Filter className="h-4 w-4" />
            <span>{total} total logs</span>
          </div>
        </div>
      </Card>

      {/* Logs Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Events</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="p-8 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary-600 mx-auto" />
              <p className="mt-2 text-dark-500">Loading audit logs...</p>
            </div>
          ) : logs.length === 0 ? (
            <div className="p-8 text-center">
              <Shield className="h-12 w-12 text-dark-200 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-dark-700">No audit logs found</h3>
              <p className="text-dark-500 mt-1">Audit logs will appear here once events occur</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-200 bg-dark-50">
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider">Event</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden md:table-cell">User</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden lg:table-cell">Resource</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden lg:table-cell">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden xl:table-cell">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-100">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-dark-50 transition-colors">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-primary-100">
                            <Shield className="h-4 w-4 text-primary-600" />
                          </div>
                          <div>
                            <p className="font-medium text-dark-900">{log.action.replace(/_/g, ' ')}</p>
                            {log.error_message && (
                              <p className="text-xs text-red-500 mt-0.5">{log.error_message}</p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 hidden md:table-cell text-sm text-dark-600">
                        {log.user_id ? `User ${log.user_id.slice(0, 8)}...` : 'System'}
                      </td>
                      <td className="px-6 py-4 hidden lg:table-cell text-sm text-dark-600">
                        {log.resource_type ? (
                          <span className="capitalize">{log.resource_type}</span>
                        ) : (
                          <span className="text-dark-400">—</span>
                        )}
                      </td>
                      <td className="px-6 py-4 hidden lg:table-cell">
                        <Badge variant={statusColors[log.success ? 'success' : 'error']}>
                          <div className="flex items-center gap-1">
                            {log.success ? (
                              <CheckCircle2 className="h-3 w-3" />
                            ) : (
                              <AlertCircle className="h-3 w-3" />
                            )}
                            {log.success ? 'Success' : 'Error'}
                          </div>
                        </Badge>
                      </td>
                      <td className="px-6 py-4 hidden xl:table-cell text-sm text-dark-500">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {format(new Date(log.created_at), 'MMM d, HH:mm:ss')}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

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
    </div>
  );
};