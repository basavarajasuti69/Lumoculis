import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-border bg-background mt-auto">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 px-4 lg:px-8 py-8">
        <div className="flex items-center gap-2 font-bold text-foreground tracking-tight">
          <span className="font-bold uppercase tracking-tighter">Lumoculis</span>
        </div>

        <nav className="flex flex-col lg:flex-row gap-4 text-sm text-secondary">
          <Link href="/" className="hover:text-accent transition-colors">
            Overview
          </Link>
          <Link href="/analyze" className="hover:text-accent transition-colors">
            Analyze Image
          </Link>
          <Link href="/model-info" className="hover:text-accent transition-colors">
            Model Information
          </Link>
          <Link href="/about" className="hover:text-accent transition-colors">
            About
          </Link>
        </nav>

        <p className="text-xs text-muted">
          Lumoculis is a research prototype, not a medical device. Not clinically validated.
        </p>

        <p className="text-xs text-muted">
          © 2026 Lumoculis
        </p>
      </div>
    </footer>
  );
}
