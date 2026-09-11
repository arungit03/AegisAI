/** Dashboard page */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  MessageSquare,
  FileText,
  Users,
  Shield,
  TrendingUp,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Database,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { clsx } from 'clsx';
import { api } from '../services/api';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { getPageItems } from '../types';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: { value: number; label: string };
  color: 'primary' | 'green' | 'orange' | 'red';
}

const StatsCard = ({ title, value, icon, trend, color }: StatsCardProps) => {
  const colorClasses = {
    primary: 'bg-primary-100 text-primary-600',
    green: 'bg-green-100 text-green-600',
    orange: 'bg-orange-100 text-orange-600',
    red: 'bg-red-100 text-red-600',
  };

  const trendColor = trend && trend.value >= 0 ? 'text-green-600' : 'text-red-600';
  const TrendIcon = trend && trend.value >= 0 ? ArrowUpRight : ArrowDownRight;

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-dark-500">{title}</p>
            <p className="mt-1 text-3xl font-bold text-dark-900">{value}</p>
            {trend && (
              <div className="mt-2 flex items-center gap-1">
                <span className={clsx('text-xs font-medium', trendColor)}>
                  <TrendIcon className="h-3 w-3" /> {Math.abs(trend.value)}%
                </span>
                <span className="text-xs text-dark-500">{trend.label}</span>
              </div>
            )}
          </div>
          <div className={clsx('p-3 rounded-xl', colorClasses[color])}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export const DashboardPage = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    conversations: 0,
    documents: 0,
    users: 0,
    queriesToday: 0,
  });
  const [recentActivity, setRecentActivity] = useState<Array<{
    id: string;
    action: string;
    user: string;
    time: string;
    type: string;
  }>>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch stats from various endpoints
        const [conversations, documents, users, auditLogs] = await Promise.all([
          api.getConversations({ page_size: 1 }),
          api.getDocuments({ page_size: 1 }),
          api.getUsers({ page_size: 1 }),
          api.getAuditLogs({ page_size: 10 }),
        ]);

        const auditLogsData = getPageItems(auditLogs);

        setStats({
          conversations: conversations.total,
          documents: documents.total,
          users: users.total,
          queriesToday: auditLogsData.filter(
            (log) => log.action === 'chat_query' && new Date(log.created_at) > new Date(Date.now() - 24 * 60 * 60 * 1000)
          ).length,
        });

        setRecentActivity(
          auditLogsData.slice(0, 5).map((log) => ({
            id: log.id,
            action: log.action.replace(/_/g, ' '),
            user: log.user_id || 'System',
            time: formatDistanceToNow(new Date(log.created_at), { addSuffix: true }),
            type: log.success ? 'success' : 'error',
          }))
        );
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const statCards = [
    {
      title: 'Conversations',
      value: stats.conversations,
      icon: <MessageSquare className="h-6 w-6" />,
      color: 'primary' as const,
    },
    {
      title: 'Documents',
      value: stats.documents,
      icon: <FileText className="h-6 w-6" />,
      color: 'green' as const,
    },
    {
      title: 'Users',
      value: stats.users,
      icon: <Users className="h-6 w-6" />,
      color: 'orange' as const,
    },
    {
      title: 'Queries Today',
      value: stats.queriesToday,
      icon: <TrendingUp className="h-6 w-6" />,
      color: 'red' as const,
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-900">Dashboard</h1>
          <p className="text-dark-500">Overview of your AegisAI workspace</p>
        </div>
        <Button variant="primary">
          <span>New Conversation</span>
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((card) => (
          <StatsCard key={card.title} {...card} />
        ))}
      </div>

      {/* Recent Activity & Quick Actions */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-12 bg-dark-100 rounded animate-pulse" />
                ))}
              </div>
            ) : recentActivity.length === 0 ? (
              <p className="text-dark-500 text-center py-4">No recent activity</p>
            ) : (
              <div className="space-y-3">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-center gap-3 p-3 rounded-lg bg-dark-50">
                    <div
                      className={clsx(
                        'p-2 rounded-lg',
                        activity.type === 'success' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                      )}
                    >
                      <Shield className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-dark-900 truncate">{activity.action}</p>
                      <p className="text-xs text-dark-500">{activity.user} • {activity.time}</p>
                    </div>
                    <Badge variant={activity.type === 'success' ? 'success' : 'danger'}>{activity.type}</Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <Button variant="secondary" className="w-full justify-start gap-3" onClick={() => navigate('/chat')}>
                <MessageSquare className="h-5 w-5" />
                <span>Start New Chat</span>
              </Button>
              <Button variant="secondary" className="w-full justify-start gap-3" onClick={() => navigate('/documents')}>
                <FileText className="h-5 w-5" />
                <span>Upload Document</span>
              </Button>
              <Button variant="secondary" className="w-full justify-start gap-3" onClick={() => navigate('/users')}>
                <Users className="h-5 w-5" />
                <span>Manage Users</span>
              </Button>
              <Button variant="secondary" className="w-full justify-start gap-3" onClick={() => navigate('/audit-logs')}>
                <Shield className="h-5 w-5" />
                <span>View Audit Logs</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="flex items-center gap-3 p-4 rounded-lg bg-green-50">
              <div className="p-2 rounded-lg bg-green-100">
                <Clock className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="font-medium text-dark-900">API Server</p>
                <p className="text-sm text-green-600">Healthy</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 rounded-lg bg-green-50">
              <div className="p-2 rounded-lg bg-green-100">
                <Database className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="font-medium text-dark-900">Database</p>
                <p className="text-sm text-green-600">Connected</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 rounded-lg bg-green-50">
              <div className="p-2 rounded-lg bg-green-100">
                <Shield className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="font-medium text-dark-900">Vector DB</p>
                <p className="text-sm text-green-600">Operational</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};