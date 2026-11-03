export function ProbabilityBar({ grade, probability, isPredicted }: { grade: number; probability: number; isPredicted?: boolean }) {
  const percent = Math.max(0, Math.min(100, probability * 100));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-foreground">Grade {grade}</span>
        <span className="text-secondary">{probability.toFixed(3)}</span>
      </div>
      <div className="w-full bg-border rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${
            isPredicted ? "bg-accent" : "bg-secondary/30"
          }`}
          style={{ width: `${percent}%` }}
        ></div>
      </div>
    </div>
  );
}
