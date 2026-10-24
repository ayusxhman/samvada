import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Samvada — AI Sign Language Translation Platform",
  description: "Real-time sign language recognition, 3D VRM avatar animations, and 50+ multilingual translations powered by Samvada.",
  manifest: "/manifest.json",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
