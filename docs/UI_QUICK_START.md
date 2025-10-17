# 🚀 Quick Start Guide - Professional UI Features

## Getting Started in 3 Steps

### Step 1: Start the Backend

```powershell
cd backend
python api/app.py
```

Backend should start on `http://localhost:8000`

### Step 2: Start the Frontend

```powershell
cd frontend
npm start
```

Frontend opens at `http://localhost:3000`

### Step 3: Use the Features!

## 📱 Feature Quick Reference

### **Severity Filters**

Click the tabs at the top of the detection panel:

- **ALL** - Shows everything
- **CRITICAL** - Red alerts only
- **HIGH** - Orange warnings
- **MEDIUM** - Yellow cautions
- **LOW** - Blue info

### **Pagination**

- Use **Prev** / **Next** buttons at the bottom
- Shows "Page X of Y"
- Auto-hides if less than 10 items

### **Clear History**

- Click **Clear** button in top-right of panel
- Confirms with backend
- Resets all detections

### **Mobile Mode**

On small screens:

- Video shows first
- Click **Show/Hide Detection Panel** button
- Panel slides up/down smoothly

### **Advanced Details**

- Click **Advanced ▾** at bottom of panel
- Shows model info, frame count, FPS
- Click again to hide

## 🎨 Visual Features

### Animations

- **New detections**: Slide in from left
- **Empty state**: Fades in gently
- **Hover cards**: Scale up slightly
- **Progress bars**: Smooth fill animation

### Color Coding

- 🚨 **Red** = CRITICAL (Score > 85%)
- ⚠️ **Orange** = HIGH (Score 75-85%)
- ⚡ **Yellow** = MEDIUM (Score 70-75%)
- ℹ️ **Blue** = LOW (Score < 70%)

### Live Indicators

- **Red pulsing dot** = Streaming active
- **Gray dot** = Idle
- **Green camera** = Connected
- **Spinning circle** = Loading history

## 🔧 Troubleshooting

### No History Loading?

1. Check backend is running: `http://localhost:8000/docs`
2. Open browser console (F12)
3. Look for fetch errors
4. Verify endpoint: `http://localhost:8000/api/detections/history`

### Panel Not Showing?

- On mobile: Click toggle button below video
- On desktop: Should always be visible
- Try refreshing page (Ctrl+F5)

### Filters Not Working?

- Check if detections have `severity` field
- Look in browser console for errors
- Try clicking "ALL" filter first

### Clear Button Missing?

- Only visible when detections exist
- Try adding a detection first
- Check `detectionHistory.length > 0`

## 💡 Pro Tips

1. **Use Filters**: Quickly find critical issues
2. **Check Advanced**: Monitor FPS and frame count
3. **Clear Regularly**: Keep panel uncluttered
4. **Mobile Friendly**: Works great on tablets
5. **Hover for Details**: Cards show more on hover

## 🎯 Keyboard Shortcuts (Future)

Coming soon:

- `C` - Clear history
- `F` - Toggle filters
- `A` - Toggle advanced
- `M` - Toggle mobile panel
- Arrow keys - Navigate pages

## 📊 API Endpoints Used

### Get History

```
GET http://localhost:8000/api/detections/history
Response: { "detections": [...] }
```

### Clear History

```
POST http://localhost:8000/api/detections/clear
Response: { "message": "Cleared X detections" }
```

### WebSocket Stream

```
WS ws://localhost:8000/ws/stream
Sends: { "type": "frame", "data": "base64..." }
Receives: { "type": "prediction", "data": {...} }
```

## 🎨 Customization Options

### Change Page Size

In `LiveCamera.js`:

```javascript
const [pageSize, setPageSize] = useState(10); // Change to 5, 20, etc.
```

### Adjust Animation Speed

In `LiveCamera.css`:

```css
.animate-slideIn {
  animation: slideIn 0.3s ease-out; /* Change 0.3s */
}
```

### Modify Colors

In component JSX, update Tailwind classes:

```javascript
// Change CRITICAL color from red to purple
className = "bg-purple-600"; // instead of bg-red-600
```

## 📈 Performance Tips

1. **Pagination**: Keep page size low (10-20) for smooth scrolling
2. **Clear Old Data**: Clear history weekly for best performance
3. **Filter Early**: Use severity filters to reduce rendered items
4. **Close Advanced**: Keep advanced panel closed when not needed
5. **Mobile**: Collapse panel when focusing on video

## 🆘 Support Checklist

Before reporting issues:

- [ ] Backend running and accessible
- [ ] Frontend compiled without errors
- [ ] Browser console checked (F12)
- [ ] Network tab shows API calls
- [ ] WebSocket connected (green indicator)
- [ ] Tried clearing browser cache
- [ ] Tested in different browser

## 📚 Related Documentation

- **Full Feature Guide**: `docs/PROFESSIONAL_UI_ENHANCEMENTS.md`
- **Fusion System**: `docs/NEW/PROFESSIONAL_FUSION_SYSTEM.md`
- **API Reference**: `backend/README.md`
- **Setup Guide**: `docs/NEW/SETUP_AND_TEST.md`

---

**Need Help?** Check browser console (F12) for error messages!
