/** Dropdown component */
import { useRef, useEffect, useState, ReactNode } from 'react';
import { ChevronDown, Check } from 'lucide-react';
import { clsx } from 'clsx';

interface DropdownOption {
  value: string;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
}

interface DropdownProps {
  options: DropdownOption[];
  value?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  triggerClassName?: string;
}

export const Dropdown = ({
  options,
  value,
  onChange,
  placeholder = 'Select...',
  className,
  disabled,
  triggerClassName,
}: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div ref={dropdownRef} className={clsx('relative inline-block w-full', className)}>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={clsx(
          'w-full flex items-center justify-between px-4 py-2.5 text-sm text-dark-900 bg-white border rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2',
          disabled
            ? 'bg-dark-50 text-dark-400 cursor-not-allowed border-dark-200'
            : 'border-dark-300 hover:border-primary-500 focus:border-primary-500 focus:ring-primary-500/20',
          triggerClassName
        )}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span>{selectedOption?.label || placeholder}</span>
        <ChevronDown
          className={clsx('h-4 w-4 text-dark-400 transition-transform', isOpen && 'rotate-180')}
          aria-hidden="true"
        />
      </button>

      {isOpen && !disabled && (
        <ul
          className="absolute z-10 mt-1 w-full max-h-60 overflow-auto rounded-lg bg-white border border-dark-200 shadow-lg py-1"
          role="listbox"
        >
          {options.map((option) => (
            <li key={option.value} role="option" aria-selected={value === option.value}>
              <button
                type="button"
                onClick={() => {
                  if (!option.disabled) {
                    onChange(option.value);
                    setIsOpen(false);
                  }
                }}
                disabled={option.disabled}
                className={clsx(
                  'w-full flex items-center gap-3 px-4 py-2 text-sm transition-colors',
                  value === option.value
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-dark-900 hover:bg-dark-50',
                  option.disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                {option.icon}
                <span className="flex-1">{option.label}</span>
                {value === option.value && <Check className="h-4 w-4 text-primary-600" />}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};