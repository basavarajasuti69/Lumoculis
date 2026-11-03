"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, Image as ImageIcon, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";

const NAV_LINKS = [
  { href: "/", label: "Overview" },
  { href: "/analyze", label: "Analyze Image" },
  { href: "/model-info", label: "Model Information" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/90">
      <div className="max-w-[1280px] mx-auto px-4 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2 font-bold text-foreground tracking-tight text-lg"
          >
            <ImageIcon className="h-6 w-6 text-accent" aria-hidden="true" />
            <span className="font-bold uppercase tracking-tighter">Lumoculis</span>
          </Link>

          <div className="flex items-center gap-4">
            <div className="hidden lg:flex gap-6">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-sm font-medium text-secondary hover:text-foreground transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>

            <Button
              asChild
              variant="default"
              size="sm"
              className="hidden lg:flex border border-border bg-card hover:bg-accent hover:text-accent-foreground"
            >
              <Link href="/analyze" className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium uppercase tracking-tighter">
                <BarChart3 className="h-4 w-4" aria-hidden="true" />
                Analyze Image
              </Link>
            </Button>

            <button
              className="lg:hidden flex items-center gap-2 rounded-md border border-border bg-card px-2 text-sm text-foreground hover:bg-accent hover:text-accent-foreground"
              onClick={() => setOpen((prev) => !prev)}
              aria-label="Open menu"
            >
              <Menu className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>

        {open && (
          <div className="lg:hidden border-b border-border bg-background p-4">
            <div className="flex flex-col gap-2">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="block rounded-md px-3 py-2 text-sm font-medium text-secondary hover:bg-accent hover:text-accent-foreground transition-colors"
                >
                  {link.label}
                </Link>
              ))}
              <Link
                href="/analyze"
                onClick={() => setOpen(false)}
                className="mt-2 block rounded-md bg-accent px-3 py-2 text-center text-xs font-bold text-accent-foreground uppercase tracking-tighter hover:bg-accent-hover transition-colors"
              >
                Analyze Image
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
