// code by jph
package ch.ethz.idsc.subare.plot;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Stroke;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartTheme;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.plot.DefaultDrawingSupplier;
import org.jfree.chart.plot.PieLabelLinkStyle;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;

import ch.ethz.idsc.tensor.img.ColorDataIndexed;
import ch.ethz.idsc.tensor.img.ColorDataLists;
import ch.ethz.idsc.tensor.sca.Clip;

public abstract class AbstractPlotBuilder {
  static {
    ChartFactory.setChartTheme(createChartTheme(false));
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

  // ---
  protected String plotLabel = "";
  protected String axisLabelX = null;
  protected String axisLabelY = null;
  protected ColorDataIndexed colorDataIndexed = ColorDataLists._097.cyclic();
  protected Stroke stroke = new BasicStroke(1f);
  protected Clip axisClipX = null;
  protected Clip axisClipY = null;

  /** Mathematica::PlotLabel
   * 
   * @param plotLabel */
  public final void setPlotLabel(String plotLabel) {
    this.plotLabel = plotLabel;
  }

  public final void setAxisLabelX(String axisLabelX) {
    this.axisLabelX = axisLabelX;
  }

  public final void setAxisLabelY(String axisLabelY) {
    this.axisLabelY = axisLabelY;
  }

  public final void setColors(ColorDataIndexed colorDataIndexed) {
    this.colorDataIndexed = colorDataIndexed;
  }

  public final void setStroke(Stroke stroke) {
    this.stroke = stroke;
  }

  public final void setAxisClipX(Clip axisClipX) {
    this.axisClipX = axisClipX;
  }

  public final void setAxisClipY(Clip axisClipY) {
    this.axisClipY = axisClipY;
  }

  /** @return new instance of JFreeChart that can be further configured by the application layer */
  public abstract JFreeChart jFreeChart();
}
