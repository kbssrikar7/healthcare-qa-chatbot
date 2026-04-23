import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://mediquery-healthcare.vercel.app"),
  title: "MediQuery AI",
  description: "Explainable Medical Question Answering driven by RAG and state-of-the-art medical knowledge.",
  openGraph: {
    title: "MediQuery AI",
    description: "Explainable Medical Question Answering Engine",
    url: "https://mediquery-healthcare.vercel.app",
    siteName: "MediQuery AI",
    images: [
      {
        url: "/opengraph-image.png",
        width: 1200,
        height: 630,
        alt: "MediQuery AI - Explainable Healthcare Knowledge Engine",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "MediQuery AI",
    description: "Explainable Medical Question Answering",
    images: ["/opengraph-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}
    >
      <body className="h-full bg-zinc-950 text-zinc-100">{children}</body>
    </html>
  );
}
