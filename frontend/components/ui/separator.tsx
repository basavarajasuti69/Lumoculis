import * as React from "react";
import { cn } from "@/lib/utils";

type SeparatorProps = React.HTMLAttributes<HTMLHRElement>;
const Separator = React.forwardRef<HTMLHRElement, SeparatorProps>(({ className, ...props }, ref) => (
  <hr
    ref={ref}
    className={cn("flex w-full border-t border-border", className)}
    {...props}
  />
));
Separator.displayName = "Separator";

export { Separator };
