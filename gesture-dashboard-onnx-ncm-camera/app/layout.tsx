import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Gesture Control Lab ONNX · 10 FPS Qualification',
  description: 'Independent ONNX Runtime deployment for MediaPipe gesture recognition, Follow Object, feedback review, and strict 10 FPS measurement.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
