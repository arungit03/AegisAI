/** Documents page */
import { useEffect, useState } from 'react';
import {
  Upload,
  Search,
  Trash2,
  Eye,
  RefreshCw,
  FileText,
  Loader2,
} from 'lucide-react';
import { format } from 'date-fns';
import { api } from '../services/api';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Card, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';
import { Dropdown } from '../components/ui/Dropdown';
import { Textarea } from '../components/ui/Textarea';
import type { Document, DocumentClassification, DocumentStatus } from '../types';
import { getPageItems } from '../types';

const classificationOptions = [
  { value: 'public_internal', label: 'Public Internal' },
  { value: 'confidential', label: 'Confidential' },
  { value: 'restricted', label: 'Restricted' },
  { value: 'highly_restricted', label: 'Highly Restricted' },
];

const statusOptions = [
  { value: 'uploaded', label: 'Uploaded' },
  { value: 'processing', label: 'Processing' },
  { value: 'processed', label: 'Processed' },
  { value: 'failed', label: 'Failed' },
  { value: 'deleted', label: 'Deleted' },
];

type BadgeVariant = 'primary' | 'success' | 'warning' | 'danger' | 'gray' | 'outline';

const classificationColors: Record<DocumentClassification, BadgeVariant> = {
  public_internal: 'gray',
  confidential: 'primary',
  restricted: 'warning',
  highly_restricted: 'danger',
};

const statusColors: Record<DocumentStatus, BadgeVariant> = {
  uploaded: 'gray',
  processing: 'primary',
  processed: 'success',
  failed: 'danger',
  deleted: 'gray',
};

export const DocumentsPage = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [classificationFilter, setClassificationFilter] = useState<DocumentClassification | ''>('');
  const [statusFilter, setStatusFilter] = useState<DocumentStatus | ''>('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const response = await api.getDocuments({
        page,
        page_size: pageSize,
        search: search || undefined,
        classification: classificationFilter || undefined,
        status: statusFilter || undefined,
      });
      setDocuments(getPageItems(response));
      setTotal(response.total);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [page, pageSize, search, classificationFilter, statusFilter]);

  const handleUpload = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const file = formData.get('file') as File;
    if (!file) return;

    setUploading(true);
    try {
      await api.uploadDocument(file, {
        title: formData.get('title') as string || undefined,
        description: formData.get('description') as string || undefined,
        classification: formData.get('classification') as DocumentClassification || undefined,
        department: formData.get('department') as string || undefined,
        tags: formData.get('tags') as string || undefined,
      });
      setShowUploadModal(false);
      (e.target as HTMLFormElement).reset();
      fetchDocuments();
    } catch (error) {
      console.error('Failed to upload document:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    try {
      await api.deleteDocument(docId);
      fetchDocuments();
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  const handleReindex = async (docId: string) => {
    try {
      await api.reindexDocument(docId);
      fetchDocuments();
    } catch (error) {
      console.error('Failed to reindex document:', error);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-900">Documents</h1>
          <p className="text-dark-500">Manage your company knowledge base</p>
        </div>
        <Button onClick={() => setShowUploadModal(true)}>
          <Upload className="h-4 w-4" />
          Upload Document
        </Button>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="relative flex-1 min-w-[200px] max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dark-400" />
            <input
              type="search"
              placeholder="Search documents..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg border border-dark-200 bg-white px-10 py-2 text-sm text-dark-900 placeholder:text-dark-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
            />
          </div>
          <Select
            options={[{ value: '', label: 'All Classifications' }, ...classificationOptions]}
            value={classificationFilter}
            onChange={(value) => setClassificationFilter(value as DocumentClassification | '')}
            placeholder="Classification"
            className="w-48"
          />
          <Select
            options={[{ value: '', label: 'All Status' }, ...statusOptions]}
            value={statusFilter}
            onChange={(value) => setStatusFilter(value as DocumentStatus | '')}
            placeholder="Status"
            className="w-40"
          />
        </div>
      </Card>

      {/* Documents Table */}
      <Card>
        <CardContent className="p-0">
          {loading ? (
            <div className="p-8 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary-600 mx-auto" />
              <p className="mt-2 text-dark-500">Loading documents...</p>
            </div>
          ) : documents.length === 0 ? (
            <div className="p-8 text-center">
              <FileText className="h-12 w-12 text-dark-200 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-dark-700">No documents found</h3>
              <p className="text-dark-500 mt-1">Upload your first document to get started</p>
              <Button className="mt-4" onClick={() => setShowUploadModal(true)}>
                <Upload className="h-4 w-4" />
                Upload Document
              </Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-200 bg-dark-50">
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider">Document</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden md:table-cell">Classification</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden lg:table-cell">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden lg:table-cell">Size</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-dark-500 uppercase tracking-wider hidden xl:table-cell">Uploaded</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-dark-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-100">
                  {documents.map((doc) => (
                    <tr key={doc.id} className="hover:bg-dark-50 transition-colors">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-primary-100">
                            <FileText className="h-5 w-5 text-primary-600" />
                          </div>
                          <div>
                            <p className="font-medium text-dark-900 truncate max-w-xs">{doc.original_filename}</p>
                            <p className="text-xs text-dark-500">{doc.title || 'No title'}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 hidden md:table-cell">
                        <Badge variant={classificationColors[doc.classification]}>
                          {doc.classification.replace('_', ' ')}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 hidden lg:table-cell">
                        <Badge variant={statusColors[doc.status]}>
                          {doc.status}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 hidden lg:table-cell text-sm text-dark-500">
                        {formatFileSize(doc.file_size)}
                      </td>
                      <td className="px-6 py-4 hidden xl:table-cell text-sm text-dark-500">
                        {format(new Date(doc.created_at), 'MMM d, yyyy')}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <Dropdown
                          options={[
                            { value: 'view', label: 'View', icon: <Eye className="h-4 w-4" /> },
                            { value: 'reindex', label: 'Re-index', icon: <RefreshCw className="h-4 w-4" /> },
                            { value: 'delete', label: 'Delete', icon: <Trash2 className="h-4 w-4" /> },
                          ]}
                          placeholder="Actions"
                          onChange={(action) => {
                            switch (action) {
                              case 'view':
                                setSelectedDocument(doc);
                                setShowDetailModal(true);
                                break;
                              case 'reindex':
                                handleReindex(doc.id);
                                break;
                              case 'delete':
                                handleDelete(doc.id);
                                break;
                            }
                          }}
                        />
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

      {/* Upload Modal */}
      <Modal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        title="Upload Document"
        size="lg"
      >
        <form onSubmit={handleUpload} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-dark-700 mb-1.5">File</label>
            <input
              type="file"
              name="file"
              accept=".pdf,.docx,.txt,.md"
              required
              className="w-full rounded-lg border border-dark-300 px-4 py-2.5 text-sm file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
            />
            <p className="mt-1 text-xs text-dark-500">PDF, DOCX, TXT, MD up to 50MB</p>
          </div>

          <Input
            name="title"
            label="Title (optional)"
            placeholder="Document title"
          />

          <div className="grid gap-4 sm:grid-cols-2">
            <Select
              name="classification"
              label="Classification"
              options={classificationOptions}
              placeholder="Public Internal"
            />
            <Input
              name="department"
              label="Department (optional)"
              placeholder="Engineering, HR, etc."
            />
          </div>

          <Input
            name="tags"
            label="Tags (optional)"
            placeholder="comma, separated, tags"
          />

          <Textarea
            name="description"
            label="Description (optional)"
            placeholder="Document description"
            rows={3}
          />

          <div className="flex justify-end gap-2 pt-4">
            <Button type="button" variant="secondary" onClick={() => setShowUploadModal(false)}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={uploading}>
              {uploading && <Loader2 className="h-4 w-4 animate-spin" />}
              Upload
            </Button>
          </div>
        </form>
      </Modal>

      {/* Document Detail Modal */}
      <Modal
        isOpen={showDetailModal}
        onClose={() => setShowDetailModal(false)}
        title={selectedDocument?.original_filename}
        size="lg"
      >
        {selectedDocument && (
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <p className="text-sm text-dark-500">Title</p>
                <p className="font-medium text-dark-900">{selectedDocument.title || 'N/A'}</p>
              </div>
              <div>
                <p className="text-sm text-dark-500">Classification</p>
                <Badge variant={classificationColors[selectedDocument.classification] as any}>
                  {selectedDocument.classification.replace('_', ' ')}
                </Badge>
              </div>
              <div>
                <p className="text-sm text-dark-500">Status</p>
                <Badge variant={statusColors[selectedDocument.status] as any}>
                  {selectedDocument.status}
                </Badge>
              </div>
              <div>
                <p className="text-sm text-dark-500">File Size</p>
                <p className="font-medium text-dark-900">{formatFileSize(selectedDocument.file_size)}</p>
              </div>
              <div>
                <p className="text-sm text-dark-500">File Type</p>
                <p className="font-medium text-dark-900">{selectedDocument.mime_type}</p>
              </div>
              <div>
                <p className="text-sm text-dark-500">Uploaded</p>
                <p className="font-medium text-dark-900">{format(new Date(selectedDocument.created_at), 'MMM d, yyyy HH:mm')}</p>
              </div>
              {selectedDocument.department && (
                <div>
                  <p className="text-sm text-dark-500">Department</p>
                  <p className="font-medium text-dark-900">{selectedDocument.department}</p>
                </div>
              )}
              {selectedDocument.tags && selectedDocument.tags.length > 0 && (
                <div>
                  <p className="text-sm text-dark-500">Tags</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedDocument.tags.map((tag) => (
                      <Badge key={tag} variant="outline">{tag}</Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {selectedDocument.description && (
              <div>
                <p className="text-sm text-dark-500">Description</p>
                <p className="text-dark-700">{selectedDocument.description}</p>
              </div>
            )}

            <div className="flex justify-end gap-2 pt-4 border-t border-dark-200">
              <Button variant="secondary" onClick={() => handleReindex(selectedDocument.id)}>
                <RefreshCw className="h-4 w-4" />
                Re-index
              </Button>
              <Button variant="danger" onClick={() => { handleDelete(selectedDocument.id); setShowDetailModal(false); }}>
                <Trash2 className="h-4 w-4" />
                Delete
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};