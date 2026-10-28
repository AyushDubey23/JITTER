import type { Metadata, Viewport } from 'next'
import { Unbounded, Instrument_Sans, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const display = Unbounded({ subsets: ['latin'], weight: ['600', '900'], variable: '--font-display' })
const body = Instrument_Sans({ subsets: ['latin'], weight: ['400', '600'], variable: '--font-body' })
const mono = JetBrains_Mono({ subsets: ['latin'], weight: ['400', '500'], variable: '--font-mono' })

export const metadata: Metadata = {
  title: 'Jitter — Root-cause analysis for on-device ML latency',
  description: 'A Python library and Android runner for diagnosing inconsistent on-device ML latency. By Ayush Dubey.',
  authors: [{ name: 'Ayush Dubey', url: 'https://github.com/AyushDubey23' }],
  creator: 'Ayush Dubey',
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/icon-dark-32x32.png', sizes: '32x32', type: 'image/png' },
      { url: '/icon.svg', type: 'image/svg+xml' },
    ],
    apple: '/apple-icon.png',
    shortcut: '/favicon.ico',
  },
  openGraph: {
    title: 'Jitter — Root-cause analysis for on-device ML latency',
    description: 'Diagnose inconsistent on-device ML latency with controlled reruns and statistical evidence. By Ayush Dubey.',
    type: 'website',
  },
}

export const viewport: Viewport = {
  colorScheme: 'dark light',
  themeColor: '#07050F',
  userScalable: true,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/icon.svg" type="image/svg+xml" />
        <link rel="apple-touch-icon" href="/apple-icon.png" />
      </head>
      <body className={`${display.variable} ${body.variable} ${mono.variable}`}>
        {children}
      </body>
    </html>
  )
}
