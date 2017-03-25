from echarts import Echart, Legend, Bar, Axis

chart = Echart('GDP', 'This is a fake chart')
chart.use(Bar('China', [2, 3, 4, 5, 1 ,2 ,3, 4, 25, 36, 27]))
chart.use(Legend(['GDP']))
chart.use(Axis('category', 'bottom', data=['Nov', 'Dec', 'Jan', 'Feb','March','April','Jun','Jul','Aug','Sep','Oct']))
chart.use(Axis('value','left', data= range(0, 100, 10 )))
chart.plot()