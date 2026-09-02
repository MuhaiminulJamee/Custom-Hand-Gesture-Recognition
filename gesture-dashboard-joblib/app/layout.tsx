import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Gesture Control Lab · 10 FPS Qualification',
  description: 'Local v18_17 MediaPipe gesture recognition, Follow Object, online feedback, and strict 10 FPS performance dashboard.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
