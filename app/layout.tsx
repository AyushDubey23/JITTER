import type { Metadata, Viewport } from 'next'
import { Unbounded, Instrument_Sans, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const display = Unbounded({ subsets: ['latin'], weight: ['600', '900'], variable: '--font-display' })
const body = Instrument_Sans({ subsets: ['latin'], weight: ['400', '600'], variable: '--font-body' })
const mono = JetBrains_Mono({ subsets: ['latin'], weight: ['400', '500'], variable: '--font-mono' })

export const metadata: Metadata = {
  title: 'Jitter — Root-cause analysis for on-device ML latency',
  description: 'A preview of a Python library and Android runner for diagnosing inconsistent on-device ML latency. Created by Ayush Dubey.',
  authors: [{ name: 'Ayush Dubey' }],
  creator: 'Ayush Dubey',
  icons: {
    icon: [
      { url: '/favicon.ico' },
      { url: '/icon.png', sizes: '512x512', type: 'image/png' },
      { url: '/icon-dark-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-icon.png', sizes: '180x180', type: 'image/png' },
    ],
  },
  openGraph: {
    title: 'Jitter — Root-cause analysis for on-device ML latency',
    description: 'Diagnosing inconsistent on-device ML latency. Created by Ayush Dubey.',
    type: 'website',
  },
}
export const viewport: Viewport = { colorScheme: 'dark light', themeColor: '#07050F', userScalable: true }
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body className={`${display.variable} ${body.variable} ${mono.variable}`}>{children}</body></html> }
