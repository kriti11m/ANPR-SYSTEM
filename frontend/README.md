# Frontend Dashboard

React.js web application providing a real-time dashboard for monitoring and managing the ANPR system.

## 🏗️ Structure

```
frontend/
├── public/            # Static assets
├── src/
│   ├── components/    # Reusable React components
│   ├── pages/         # Page components
│   ├── hooks/         # Custom React hooks
│   ├── utils/         # Utility functions
│   ├── services/      # API service functions
│   ├── styles/        # CSS/SCSS files
│   └── App.js         # Main application component
├── package.json       # Node.js dependencies
└── public/            # Static files
```

## 🚀 Features

- **Real-time Dashboard**: Live video feeds and detection results
- **Historical Data**: Browse and search past detections
- **Feed Management**: Configure and monitor video sources
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark/Light Theme**: Customizable user interface
- **Export Functionality**: Download detection data and reports

## 🛠️ Setup

### Local Development
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

### Environment Variables
Create a `.env` file in the frontend directory:
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_VERSION=1.0.0
```

## 🧩 Components

### Layout Components
- `Header` - Navigation and user menu
- `Sidebar` - Main navigation menu
- `Footer` - Application footer
- `Layout` - Main layout wrapper

### Dashboard Components
- `LiveFeed` - Real-time video feed display
- `DetectionCard` - Individual detection result
- `MetricsWidget` - System metrics display
- `AlertPanel` - System alerts and notifications

### Data Components
- `DetectionTable` - Tabular view of detections
- `SearchFilters` - Search and filter controls
- `ExportButton` - Data export functionality
- `Pagination` - Table pagination controls

## 📱 Pages

### Dashboard (`/`)
- Real-time feed monitoring
- Recent detection results
- System health metrics
- Quick action buttons

### Detections (`/detections`)
- Comprehensive detection history
- Advanced search and filtering
- Detailed detection view modal
- Export and download options

### Feeds (`/feeds`)
- Video feed management
- Add/edit/delete feed sources
- Feed status monitoring
- Configuration settings

### Settings (`/settings`)
- User preferences
- System configuration
- Theme selection
- Notification settings

### Reports (`/reports`)
- Detection analytics
- Performance metrics
- Custom report generation
- Data visualization charts

## 🎨 Styling

The application uses:
- **CSS Modules** - Component-scoped styling
- **Material-UI** - React component library
- **Styled Components** - CSS-in-JS styling
- **Responsive Grid** - Mobile-first design

### Theme Configuration
```javascript
const theme = {
  colors: {
    primary: '#1976d2',
    secondary: '#dc004e',
    background: '#f5f5f5',
    surface: '#ffffff',
    text: '#333333'
  },
  breakpoints: {
    mobile: '480px',
    tablet: '768px',
    desktop: '1024px'
  }
};
```

## 🔗 API Integration

### Services
- `apiService.js` - HTTP client configuration
- `detectionService.js` - Detection-related API calls
- `feedService.js` - Feed management API calls
- `authService.js` - Authentication API calls

### WebSocket Integration
```javascript
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };
    
    return () => ws.close();
  }, [url]);
  
  return data;
};
```

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

### Testing Libraries
- Jest - Testing framework
- React Testing Library - Component testing
- MSW - API mocking
- Cypress - E2E testing

## 🚀 Deployment

### Production Build
```bash
npm run build
```

### Docker Deployment
```dockerfile
FROM node:16-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
```

### Environment-specific Builds
- Development: `npm start`
- Staging: `npm run build:staging`
- Production: `npm run build:production`

## 🔧 Configuration

### Build Configuration
- Webpack configuration in `craco.config.js`
- ESLint rules in `.eslintrc.js`
- Prettier configuration in `.prettierrc`
- TypeScript configuration in `tsconfig.json`

### Performance Optimization
- Code splitting with React.lazy()
- Image optimization and lazy loading
- Bundle analysis with webpack-bundle-analyzer
- Service worker for caching

## 📊 State Management

Using Redux Toolkit for state management:
```javascript
const store = configureStore({
  reducer: {
    detections: detectionsSlice.reducer,
    feeds: feedsSlice.reducer,
    auth: authSlice.reducer,
    ui: uiSlice.reducer
  }
});
```

## 🛠️ Development Tools

- **ESLint** - Code linting
- **Prettier** - Code formatting
- **Husky** - Git hooks
- **Storybook** - Component development
- **React DevTools** - Development debugging

## 📱 Progressive Web App

The frontend is configured as a PWA with:
- Service worker for offline functionality
- Web app manifest for installation
- Push notifications support
- Background sync capabilities
