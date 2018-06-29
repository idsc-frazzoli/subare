// code by fluric
package ch.ethz.idsc.subare.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.DefaultDrawingSupplier;
import org.jfree.chart.plot.PieLabelLinkStyle;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer3D;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.title.LegendTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RectangleInsets;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.img.ColorDataLists;

// TODO class contains non generic functionality, e.g. "Number episodes"
public enum PlotUtils {
  ;
  private static final Color COLOR_BACKGROUND_PAINT = Color.WHITE;
  private static final Color COLOR_GRIDLINE_PAINT = Color.LIGHT_GRAY;
  private static final int WIDTH = 1280;
  private static final int HEIGHT = 720;

  private static File directory() {
    File directory = UserHome.Pictures("plots");
    directory.mkdir();
    return directory;
  }

  /** A plot can be created and is stored at {@code plot/path}. XYs contains a list of tensors (data sets) that again
   * contains tensors with x and y components. The list of names contains the data set names.
   * @param XYs
   * @param names */
  public static void createPlot(List<Tensor> XYs, List<String> names, String path) {
    // create plot
    final XYDataset data1 = createDataset(XYs, names);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number episodes", "Error", //
          data1, directory(), //
          false, 100, 500, false, 100, 500);
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path) {
    // create plot
    final XYDataset data1 = createDataset(map);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number episodes", "Error", //
          data1, directory(), //
          false, 100, 500, false, 100, 500);
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  private static XYDataset createDataset(Map<String, Tensor> map) {
    final XYSeriesCollection collection = new XYSeriesCollection();
    // create dataset
    for (Entry<String, Tensor> entry : map.entrySet()) {
      XYSeries series = new XYSeries(entry.getKey());
      Tensor XY = entry.getValue();
      for (int j = 0; j < XY.length(); ++j) {
        series.add(XY.get(j).Get(0).number(), XY.get(j).Get(1).number());
      }
      collection.addSeries(series);
    }
    return collection;
  }

  private static XYDataset createDataset(List<Tensor> XYs, List<String> names) {
    final XYSeriesCollection collection = new XYSeriesCollection();
    // create dataset
    for (int i = 0; i < names.size(); ++i) {
      XYSeries series = new XYSeries(names.get(i));
      Tensor XY = XYs.get(i);
      for (int j = 0; j < XY.length(); ++j) {
        series.add(XY.get(j).Get(0).number(), XY.get(j).Get(1).number());
      }
      collection.addSeries(series);
    }
    return collection;
  }

  private static File plot( //
      String filename, String diagramTitle, String axisLabelX, String axisLabelY, XYDataset dataset, //
      File directory, //
      boolean setYAxis, double minYValue, double maxYValue, boolean setXAxis, double minXValue, double maxXValue) throws Exception {
    ChartFactory.setChartTheme(getChartTheme(false));
    JFreeChart chart = ChartFactory.createXYLineChart(diagramTitle, axisLabelX, axisLabelY, dataset, PlotOrientation.VERTICAL, false, false, false); //
    for (int k = 0; k < dataset.getSeriesCount(); k++) {
      chart.getXYPlot().getRenderer().setSeriesStroke(k, new BasicStroke(2.0f));
      chart.getXYPlot().getRenderer().setSeriesPaint(k, ColorDataLists._097.cyclic().getColor(k));
    }
    chart.getXYPlot().setRangeGridlinePaint(Color.LIGHT_GRAY);
    chart.getXYPlot().setDomainGridlinePaint(Color.LIGHT_GRAY);
    chart.getPlot().setBackgroundPaint(COLOR_BACKGROUND_PAINT);
    chart.getXYPlot().setRangeGridlinePaint(COLOR_GRIDLINE_PAINT);
    chart.getXYPlot().getDomainAxis().setLowerMargin(0.0);
    chart.getXYPlot().getDomainAxis().setUpperMargin(0.0);
    if (setYAxis) {
      NumberAxis rangeAxis = (NumberAxis) chart.getXYPlot().getRangeAxis();
      rangeAxis.setRange(minYValue, maxYValue);
    }
    if (setXAxis) {
      NumberAxis rangeAxis = (NumberAxis) chart.getXYPlot().getDomainAxis();
      rangeAxis.setRange(minXValue, maxXValue);
    }
    LegendTitle legend = new LegendTitle(chart.getXYPlot().getRenderer());
    legend.setPosition(RectangleEdge.TOP);
    chart.addLegend(legend);
    return savePlot(directory, filename, chart);
  }

  private static File savePlot(File directory, String fileTitle, JFreeChart chart) throws Exception {
    File fileChart = new File(directory, fileTitle + ".png");
    ChartUtilities.saveChartAsPNG(fileChart, chart, WIDTH, HEIGHT);
    GlobalAssert.that(fileChart.isFile());
    System.out.println("Exported " + fileTitle + ".png");
    return fileChart;
  }

  private static StandardChartTheme getChartTheme(boolean shadow) {
    StandardChartTheme theme = new StandardChartTheme("amodeus");
    theme.setExtraLargeFont(new Font("Dialog", Font.BOLD, 24));
    theme.setLargeFont(new Font("Dialog", Font.PLAIN, 18));
    theme.setRegularFont(new Font("Dialog", Font.PLAIN, 14));
    theme.setSmallFont(new Font("Dialog", Font.PLAIN, 10));
    theme.setTitlePaint(Color.BLACK);
    theme.setSubtitlePaint(Color.BLACK);
    theme.setLegendBackgroundPaint(Color.WHITE);
    theme.setLegendItemPaint(Color.BLACK);
    theme.setChartBackgroundPaint(Color.WHITE);
    theme.setDrawingSupplier(new DefaultDrawingSupplier());
    theme.setPlotBackgroundPaint(Color.WHITE);
    theme.setPlotOutlinePaint(Color.BLACK);
    theme.setLabelLinkStyle(PieLabelLinkStyle.STANDARD);
    theme.setAxisOffset(new RectangleInsets(4, 4, 4, 4));
    theme.setDomainGridlinePaint(Color.WHITE);
    theme.setRangeGridlinePaint(Color.WHITE);
    theme.setBaselinePaint(Color.BLACK);
    theme.setCrosshairPaint(Color.BLACK);
    theme.setAxisLabelPaint(Color.DARK_GRAY);
    theme.setTickLabelPaint(Color.DARK_GRAY);
    theme.setBarPainter(new StandardBarPainter());
    theme.setXYBarPainter(new StandardXYBarPainter());
    theme.setShadowVisible(shadow);
    theme.setItemLabelPaint(Color.BLACK);
    theme.setThermometerPaint(Color.WHITE);
    theme.setWallPaint(BarRenderer3D.DEFAULT_WALL_PAINT);
    theme.setErrorIndicatorPaint(Color.RED);
    return theme;
  }
}
