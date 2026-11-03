import * as React from "react";
import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "secondary" | "destructive" | "outline" | "success" | "warning" | "accent";
  size?: "default" | "sm" | "lg";
}

const sizeVariants = {
  default: "px-2.5 py-0.5 text-xs",
  sm: "px-2 py-0.5 text-[0.7rem]",
  lg: "px-3 py-1 text-sm",
} as const;

const variantVariants = {
  default: "border-border text-foreground",
  secondary: "border-border text-secondary",
  destructive: "border-destructive/50 text-destructive dark:border-destructive dark:text-destructive/90",
  outline: "border-border text-foreground",
  success: "border-emerald-500/30 text-emerald-400 bg-emerald-500/10",
  warning: "border-amber-500/30 text-amber-400 bg-amber-500/10",
  accent: "border-accent/30 text-accent bg-accent/10",
} as const;

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(({ className, variant = "default", size = "default", ...props }, ref) => {
  return (
    <span
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variantVariants[variant],
        sizeVariants[size],
        className
      )}
      {...props}
    />
  );
});
Badge.displayName = "Badge";

export { Badge };
