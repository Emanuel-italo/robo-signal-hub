import { useEffect, useRef, useState, useCallback } from 'react';
import { toast } from 'sonner';

export function useWakeLock() {
  const [isLocked, setIsLocked] = useState(false);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  const requestLock = useCallback(async () => {
    // Verifica se o navegador suporta essa função
    if ('wakeLock' in navigator) {
      try {
        const lock = await navigator.wakeLock.request('screen');
        wakeLockRef.current = lock;
        setIsLocked(true);
        
        lock.addEventListener('release', () => {
          setIsLocked(false);
        });
        
        // Feedback visual opcional (pode comentar se achar chato)
        // toast.success("Modo Tela Sempre Ligada: ATIVO 💡");
      } catch (err) {
        console.error('Erro ao ativar Wake Lock:', err);
      }
    } else {
      console.warn('Seu navegador não suporta Wake Lock API');
    }
  }, []);

  const releaseLock = useCallback(async () => {
    if (wakeLockRef.current) {
      try {
        await wakeLockRef.current.release();
        wakeLockRef.current = null;
        setIsLocked(false);
      } catch (err) {
        console.error('Erro ao liberar Wake Lock:', err);
      }
    }
  }, []);

  // Reativa o bloqueio se a aba ficar visível novamente
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && isLocked) {
        requestLock();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (wakeLockRef.current) wakeLockRef.current.release();
    };
  }, [isLocked, requestLock]);

  return { isLocked, requestLock, releaseLock };
}