import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
}

export function FeatureCard({ title, description, icon }: FeatureCardProps) {
  return (
    <Card className="h-full border-border bg-card group">
      <CardHeader>
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-accent/10 p-2 text-accent group-hover:bg-accent group-hover:text-accent-foreground transition-colors">
            {icon}
          </span>
        </div>
        <CardTitle className="text-lg font-bold leading-tight">{title}</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <p className="text-sm text-secondary leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  );
}
