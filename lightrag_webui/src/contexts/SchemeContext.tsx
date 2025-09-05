// contexts/SchemeContext.tsx
import React, { createContext, useContext, useState } from 'react';
import { Scheme } from '@/api/lightrag';

interface SchemeContextType {
  selectedScheme: Scheme | undefined;
  setSelectedScheme: (scheme: Scheme | undefined) => void;
}

const SchemeContext = createContext<SchemeContextType | undefined>(undefined);

export const SchemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedScheme, setSelectedScheme] = useState<Scheme | undefined>();

  return (
    <SchemeContext.Provider value={{ selectedScheme, setSelectedScheme }}>
      {children}
    </SchemeContext.Provider>
  );
};

export const useScheme = () => {
  const context = useContext(SchemeContext);
  if (context === undefined) {
    throw new Error('useScheme must be used within a SchemeProvider');
  }
  return context;
};
