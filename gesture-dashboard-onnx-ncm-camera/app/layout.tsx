import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Gesture Control Lab · Eight-Gesture ONNX',
  description: 'Ten-FPS NCM camera gesture control with eight commands, corrected Left/Right semantics, open-set rejection, stabilized landmarks, and validation-gated feedback learning.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
