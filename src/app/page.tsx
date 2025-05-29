"use client"

import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table"

export default function Dashboard() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 flex">
            <a className="mr-6 flex items-center space-x-2" href="/">
              <span className="font-bold">🚢 SeaLogix</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container flex-1 space-y-4 p-8 pt-6">
        <div className="flex items-center justify-between space-y-2">
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <div className="flex items-center space-x-2">
            <Button>Download Report</Button>
          </div>
        </div>

        {/* Overview Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card className="p-4">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">Total Ships</p>
              <h3 className="text-2xl font-bold">24</h3>
            </div>
          </Card>
          <Card className="p-4">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">Active Alerts</p>
              <h3 className="text-2xl font-bold text-red-600">7</h3>
            </div>
          </Card>
          <Card className="p-4">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">Maintenance Tasks</p>
              <h3 className="text-2xl font-bold">156</h3>
            </div>
          </Card>
          <Card className="p-4">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">System Health</p>
              <h3 className="text-2xl font-bold text-green-600">98%</h3>
            </div>
          </Card>
        </div>

        {/* Main Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
            <TabsTrigger value="maintenance">Maintenance</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-4">
            {/* Recent Alerts */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-4">
                <div className="p-6">
                  <h4 className="text-xl font-semibold">Recent Alerts</h4>
                  <ScrollArea className="h-[400px] mt-4">
                    <div className="space-y-4">
                      <Alert>
                        <AlertTitle>High Engine Temperature</AlertTitle>
                        <AlertDescription>
                          Ship ID: VSL001 - Temperature exceeded threshold at 14:30
                        </AlertDescription>
                      </Alert>
                      <Alert>
                        <AlertTitle>Fuel Level Warning</AlertTitle>
                        <AlertDescription>
                          Ship ID: VSL003 - Fuel level below 20% threshold
                        </AlertDescription>
                      </Alert>
                      <Alert>
                        <AlertTitle>Maintenance Due</AlertTitle>
                        <AlertDescription>
                          Ship ID: VSL007 - Scheduled maintenance required
                        </AlertDescription>
                      </Alert>
                    </div>
                  </ScrollArea>
                </div>
              </Card>

              {/* Ship Status */}
              <Card className="col-span-3">
                <div className="p-6">
                  <h4 className="text-xl font-semibold mb-4">Ship Status</h4>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Ship ID</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell>VSL001</TableCell>
                        <TableCell className="text-yellow-600">Warning</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>VSL002</TableCell>
                        <TableCell className="text-green-600">Normal</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>VSL003</TableCell>
                        <TableCell className="text-red-600">Alert</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="alerts" className="space-y-4">
            {/* Alerts content will be implemented */}
            <Card className="p-6">
              <h4 className="text-xl font-semibold">Alert Management</h4>
              <p className="text-muted-foreground mt-2">Coming soon...</p>
            </Card>
          </TabsContent>

          <TabsContent value="maintenance" className="space-y-4">
            {/* Maintenance content will be implemented */}
            <Card className="p-6">
              <h4 className="text-xl font-semibold">Maintenance Schedule</h4>
              <p className="text-muted-foreground mt-2">Coming soon...</p>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            {/* Analytics content will be implemented */}
            <Card className="p-6">
              <h4 className="text-xl font-semibold">Performance Analytics</h4>
              <p className="text-muted-foreground mt-2">Coming soon...</p>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
