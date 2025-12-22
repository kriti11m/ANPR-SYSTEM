import { configureStore } from '@reduxjs/toolkit';
import authSlice from './slices/authSlice';
import detectionsSlice from './slices/detectionsSlice';
import feedsSlice from './slices/feedsSlice';
import uiSlice from './slices/uiSlice';

export const store = configureStore({
  reducer: {
    auth: authSlice,
    detections: detectionsSlice,
    feeds: feedsSlice,
    ui: uiSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
