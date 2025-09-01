import React, { useRef,useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog';
import Button from "@/components/ui/Button";
import { PlusIcon } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/Alert";
import { AlertCircle } from "lucide-react";
import {
  getSchemes,
  saveSchemes,
  addScheme,
  deleteScheme,
  Scheme
} from '@/api/lightrag';
import { useScheme } from '@/contexts/SchemeContext';
import { useTranslation } from 'react-i18next';

interface SchemeConfig {
  framework: 'lightrag' | 'raganything';
  extractor?: 'mineru' | 'docling';
}

const SchemeManagerDialog = () => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [schemes, setSchemes] = useState<Scheme[]>([]);
  const [newSchemeName, setNewSchemeName] = useState("");
  const [error, _setError] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(true);
  const setError = (err?: string) => _setError(err);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { selectedScheme, setSelectedScheme } = useScheme();

  // 加载方案数据
  useEffect(() => {
    const loadSchemes = async () => {
      try {
        setIsLoading(true);
        const response = await getSchemes();
        setSchemes(response.data);
        localStorage.getItem('selectedSchemeId') && setSelectedScheme(response.data.find(s => s.id === Number(localStorage.getItem('selectedSchemeId'))) || undefined);
      } catch (err) {
        setError(err instanceof Error ? err.message : t('schemeManager.errors.loadFailed'));
      } finally {
        setIsLoading(false);
      }
    };
    loadSchemes();
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    handleSelectScheme(selectedScheme?.id!);
    if (!scrollRef.current) return;
    const scrollToBottom = () => {
      const container = scrollRef.current!;
      const { scrollHeight } = container;
      container.scrollTop = scrollHeight;
    };
    setTimeout(scrollToBottom, 0);
  }, [schemes]);

  // 检查方案名是否已存在
  const isNameTaken = (name: string): boolean => {
    return schemes.some(scheme => scheme.name.trim() === name.trim());
  };

  // 选中方案（更新 Context）
  const handleSelectScheme = (schemeId: number) => {
    const scheme = schemes.find((s) => s.id === schemeId);
    if (scheme) {
      setSelectedScheme(scheme);
      localStorage.setItem('selectedSchemeId', String(scheme.id));
    }
  };

  // 添加新方案
  const handleAddScheme = async () => {
    const trimmedName = newSchemeName.trim();
    if (!trimmedName) {
      setError(t('schemeManager.errors.nameEmpty'));
      return;
    }
    if (isNameTaken(trimmedName)) {
      setError(t('schemeManager.errors.nameExists'));
      return;
    }

    try {
      const newScheme = await addScheme({
        name: trimmedName,
        config: { framework: 'lightrag', extractor: undefined },
      });

      // 更新方案列表
      setSchemes((prevSchemes) => [...prevSchemes, newScheme]);

      // 选中新方案
      setSelectedScheme(newScheme);

      // 清空输入和错误
      setNewSchemeName("");
      setError(undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('schemeManager.errors.addFailed'));
    }
  };

  // 删除方案
  const handleDeleteScheme = async (schemeId: number) => {
    try {
      await deleteScheme(schemeId);
      setSchemes(schemes.filter(s => s.id !== schemeId));
      if (selectedScheme?.id === schemeId) {
        setSelectedScheme(undefined); // 清除 Context 中的选中状态
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t('schemeManager.errors.deleteFailed'));
    }
  };

  // 更新方案配置
  const handleConfigChange = async (updates: Partial<SchemeConfig>) => {
    if (!selectedScheme) return;

    const updatedScheme = {
      ...selectedScheme,
      config: {
        ...selectedScheme.config,
        ...updates,
        framework: updates.framework ?? selectedScheme.config?.framework ?? 'lightrag',
        extractor: updates.extractor || selectedScheme.config?.extractor || (updates.framework === 'raganything' ? 'mineru' : undefined),
      },
    };

    setSchemes(schemes.map(s => s.id === selectedScheme.id ? updatedScheme : s));
    await saveSchemes([updatedScheme]);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="default" side="bottom" size="sm">
          <PlusIcon className="size-4" />
          {t('schemeManager.button')}
        </Button>
      </DialogTrigger>

      <DialogContent className="sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{t('schemeManager.title')}</DialogTitle>
          <DialogDescription>{t('schemeManager.description')}</DialogDescription>
        </DialogHeader>

        <div className="flex h-[500px] gap-4">
          {/* 左侧：方案列表 */}
          <div className="w-1/3 rounded-lg border p-4 bg-gray-50 flex flex-col">
            <h3 className="mb-4 font-semibold">{t('schemeManager.schemeList')}</h3>

            {/* 创建新方案输入框 */}
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={newSchemeName}
                onChange={(e) => {
                  if (e.target.value.length > 50) return;
                  setNewSchemeName(e.target.value);
                  setError(undefined);
                }}
                onKeyPress={(e) => e.key === 'Enter' && handleAddScheme()}
                placeholder={t('schemeManager.inputPlaceholder')}
                className="w-full px-3 py-1.5 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <Button onClick={handleAddScheme} size="sm">
                <PlusIcon className="size-4" />
              </Button>
            </div>

            {/* 错误提示 */}
            {error && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="size-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* 方案列表 */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto border rounded-md p-1 bg-white">
              {schemes.length === 0 ? (
                <p className="text-gray-500 text-center py-4">{t('schemeManager.emptySchemes')}</p>
              ) : (
                <div className="space-y-2">
                  {schemes.map((scheme) => (
                    <div
                      key={scheme.id}
                      className={`flex items-center justify-between p-2 rounded-md cursor-pointer transition-colors truncate ${
                        selectedScheme?.id === scheme.id
                          ? "bg-blue-100 text-blue-700"
                          : "hover:bg-gray-100"
                      }`}
                      onClick={() => handleSelectScheme(scheme.id)}
                    >
                      <div className="flex-1 truncate mr-2" title={scheme.name}>
                        {scheme.name}
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteScheme(scheme.id);
                        }}
                        className="ml-2 text-red-500 hover:text-red-700 hover:bg-red-100 rounded-full p-1 transition-colors"
                        title={t('schemeManager.deleteTooltip')}
                      >
                        −
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 右侧：方案配置 */}
          <div className="flex-1 rounded-lg border p-4 bg-gray-50">
            <h3 className="mb-4 font-semibold">{t('schemeManager.schemeConfig')}</h3>

            {selectedScheme ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm mb-1">{t('schemeManager.processingFramework')}</label>
                  <select
                    value={selectedScheme.config?.framework || "lightrag"}
                    onChange={(e) => handleConfigChange({ framework: e.target.value as 'lightrag' | 'raganything' })}
                    className="w-full px-3 py-1.5 border rounded-md focus:outline-none"
                  >
                    <option value="lightrag">LightRAG</option>
                    <option value="raganything">RAGAnything</option>
                  </select>
                </div>

                {selectedScheme.config?.framework === "raganything" && (
                  <div>
                    <label className="block text-sm mb-1">{t('schemeManager.extractionTool')}</label>
                    <select
                      value={selectedScheme.config?.extractor || "mineru"}
                      onChange={(e) => handleConfigChange({ extractor: e.target.value as 'mineru' | 'docling' })}
                      className="w-full px-3 py-1.5 border rounded-md focus:outline-none"
                    >
                      <option value="mineru">Mineru</option>
                      <option value="docling">DocLing</option>
                    </select>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[70%] text-gray-500">
                <AlertCircle className="size-12 mb-4 opacity-50" />
                <p>{t('schemeManager.selectSchemePrompt')}</p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SchemeManagerDialog;
