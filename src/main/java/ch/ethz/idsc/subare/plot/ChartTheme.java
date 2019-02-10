/* amodeus - Copyright (c) 2018, ETH Zurich, Institute for Dynamic Systems and Control */
package ch.ethz.idsc.subare.plot;

import java.awt.Color;
import java.awt.Font;

import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.plot.DefaultDrawingSupplier;
import org.jfree.chart.plot.PieLabelLinkStyle;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;

/* package */ class ChartTheme {
  private static final Color TRANSPARENT = new Color(0, 0, 0, 0);

  private static StandardChartTheme getChartTheme(StandardChartTheme standardChartTheme) {
    standardChartTheme.setExtraLargeFont(new Font(Font.DIALOG, Font.BOLD, 24));
    standardChartTheme.setLargeFont(new Font(Font.DIALOG, Font.PLAIN, 18));
    standardChartTheme.setRegularFont(new Font(Font.DIALOG, Font.PLAIN, 14));
    standardChartTheme.setSmallFont(new Font(Font.DIALOG, Font.PLAIN, 10));
    // ---
    standardChartTheme.setChartBackgroundPaint(TRANSPARENT);
    standardChartTheme.setPlotBackgroundPaint(TRANSPARENT);
    standardChartTheme.setLegendBackgroundPaint(TRANSPARENT);
    // ---
    standardChartTheme.setTitlePaint(Color.BLACK);
    standardChartTheme.setSubtitlePaint(Color.BLACK);
    standardChartTheme.setLegendItemPaint(Color.BLACK);
    // ---
    standardChartTheme.setDrawingSupplier(new DefaultDrawingSupplier());
    standardChartTheme.setPlotOutlinePaint(Color.BLACK);
    standardChartTheme.setLabelLinkStyle(PieLabelLinkStyle.STANDARD);
    standardChartTheme.setDomainGridlinePaint(Color.LIGHT_GRAY);
    standardChartTheme.setRangeGridlinePaint(Color.LIGHT_GRAY);
    standardChartTheme.setBaselinePaint(Color.BLACK);
    standardChartTheme.setCrosshairPaint(Color.BLACK);
    standardChartTheme.setAxisLabelPaint(Color.DARK_GRAY);
    standardChartTheme.setTickLabelPaint(Color.DARK_GRAY);
    standardChartTheme.setBarPainter(new StandardBarPainter());
    standardChartTheme.setXYBarPainter(new StandardXYBarPainter());
    standardChartTheme.setItemLabelPaint(Color.BLACK);
    standardChartTheme.setThermometerPaint(Color.WHITE);
    standardChartTheme.setErrorIndicatorPaint(Color.RED);
    return standardChartTheme;
  }

  public static final StandardChartTheme STANDARD = getChartTheme(new StandardChartTheme("subare"));
}
