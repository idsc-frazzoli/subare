// code by jph, fluric
package ch.ethz.idsc.subare.plot;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Stroke;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartTheme;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.DefaultDrawingSupplier;
import org.jfree.chart.plot.PieLabelLinkStyle;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.title.LegendTitle;
import org.jfree.chart.ui.RectangleEdge;
import org.jfree.data.xy.XYDataset;

import ch.ethz.idsc.tensor.img.ColorDataIndexed;
import ch.ethz.idsc.tensor.img.ColorDataLists;
import ch.ethz.idsc.tensor.sca.Clip;

public class ListPlotBuilder {
  static {
    ChartFactory.setChartTheme(createChartTheme(false));
  }
  private static final Color COLOR_BACKGROUND_PAINT = Color.WHITE;
  private static final Color COLOR_GRIDLINE_PAINT = Color.LIGHT_GRAY;
  // ---
  private final XYDataset xyDataset;
  private final JFreeChart jFreeChart;

  public ListPlotBuilder(String diagramTitle, String axisLabelX, String axisLabelY, XYDataset xyDataset) {
    this.xyDataset = xyDataset;
    jFreeChart = ChartFactory.createXYLineChart( //
        diagramTitle, axisLabelX, axisLabelY, xyDataset, PlotOrientation.VERTICAL, false, false, false); //
    setColors(ColorDataLists._097.cyclic());
    setStroke(new BasicStroke(2.0f));
    jFreeChart.getXYPlot().setRangeGridlinePaint(Color.LIGHT_GRAY);
    jFreeChart.getXYPlot().setDomainGridlinePaint(Color.LIGHT_GRAY);
    jFreeChart.getPlot().setBackgroundPaint(COLOR_BACKGROUND_PAINT);
    jFreeChart.getXYPlot().setRangeGridlinePaint(COLOR_GRIDLINE_PAINT);
    jFreeChart.getXYPlot().getDomainAxis().setLowerMargin(0.0);
    jFreeChart.getXYPlot().getDomainAxis().setUpperMargin(0.0);
    LegendTitle legendTitle = new LegendTitle(jFreeChart.getXYPlot().getRenderer());
    legendTitle.setPosition(RectangleEdge.TOP);
    jFreeChart.addLegend(legendTitle);
  }

  public JFreeChart getJFreeChart() {
    return jFreeChart;
  }

  public void setColors(ColorDataIndexed colorDataIndexed) {
    for (int k = 0; k < xyDataset.getSeriesCount(); ++k)
      jFreeChart.getXYPlot().getRenderer().setSeriesPaint(k, colorDataIndexed.getColor(k));
  }

  public void setStroke(Stroke stroke) {
    for (int k = 0; k < xyDataset.getSeriesCount(); ++k)
      jFreeChart.getXYPlot().getRenderer().setSeriesStroke(k, stroke);
  }

  public void setXAxis(Clip clip) {
    NumberAxis numberAxis = (NumberAxis) jFreeChart.getXYPlot().getDomainAxis();
    numberAxis.setRange(clip.min().number().doubleValue(), clip.max().number().doubleValue());
  }

  public void setYAxis(Clip clip) {
    NumberAxis numberAxis = (NumberAxis) jFreeChart.getXYPlot().getRangeAxis();
    numberAxis.setRange(clip.min().number().doubleValue(), clip.max().number().doubleValue());
  }

  private static ChartTheme createChartTheme(boolean shadow) {
    StandardChartTheme standardChartTheme = new StandardChartTheme("idsc");
    standardChartTheme.setExtraLargeFont(new Font(Font.DIALOG, Font.BOLD, 24));
    standardChartTheme.setLargeFont(new Font(Font.DIALOG, Font.PLAIN, 18));
    standardChartTheme.setRegularFont(new Font(Font.DIALOG, Font.PLAIN, 14));
    standardChartTheme.setSmallFont(new Font(Font.DIALOG, Font.PLAIN, 10));
    standardChartTheme.setTitlePaint(Color.BLACK);
    standardChartTheme.setSubtitlePaint(Color.BLACK);
    standardChartTheme.setLegendBackgroundPaint(Color.WHITE);
    standardChartTheme.setLegendItemPaint(Color.BLACK);
    standardChartTheme.setChartBackgroundPaint(Color.WHITE);
    standardChartTheme.setDrawingSupplier(new DefaultDrawingSupplier());
    standardChartTheme.setPlotBackgroundPaint(Color.WHITE);
    standardChartTheme.setPlotOutlinePaint(Color.BLACK);
    standardChartTheme.setLabelLinkStyle(PieLabelLinkStyle.STANDARD);
    // standardChartTheme.setAxisOffset(new RectangleInsets(4, 4, 4, 4));
    standardChartTheme.setDomainGridlinePaint(Color.WHITE);
    standardChartTheme.setRangeGridlinePaint(Color.WHITE);
    standardChartTheme.setBaselinePaint(Color.BLACK);
    standardChartTheme.setCrosshairPaint(Color.BLACK);
    standardChartTheme.setAxisLabelPaint(Color.DARK_GRAY);
    standardChartTheme.setTickLabelPaint(Color.DARK_GRAY);
    standardChartTheme.setBarPainter(new StandardBarPainter());
    standardChartTheme.setXYBarPainter(new StandardXYBarPainter());
    standardChartTheme.setShadowVisible(shadow);
    standardChartTheme.setItemLabelPaint(Color.BLACK);
    standardChartTheme.setThermometerPaint(Color.WHITE);
    // standardChartTheme.setWallPaint(BarRenderer3D.DEFAULT_WALL_PAINT);
    standardChartTheme.setErrorIndicatorPaint(Color.RED);
    return standardChartTheme;
  }
}
