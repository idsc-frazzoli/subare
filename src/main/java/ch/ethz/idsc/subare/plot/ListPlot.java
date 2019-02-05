// code by jph, fluric
package ch.ethz.idsc.subare.plot;

import java.awt.BasicStroke;
import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.title.LegendTitle;
import org.jfree.chart.ui.RectangleEdge;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ListPlot.html">ListPlot</a> */
public class ListPlot extends AbstractPlotBuilder {
  private final XYSeriesList xySeriesList;

  public ListPlot(XYSeriesList xySeriesList) {
    this.xySeriesList = xySeriesList;
    setStroke(new BasicStroke(2f));
  }

  @Override
  public JFreeChart jFreeChart() {
    JFreeChart jFreeChart = ChartFactory.createXYLineChart( //
        plotLabel, axisLabelX, axisLabelY, xySeriesList.xySeriesCollection(), PlotOrientation.VERTICAL, //
        false, // legend
        false, // tooltips
        false); // urls
    final XYPlot xyPlot = jFreeChart.getXYPlot();
    final XYItemRenderer xyItemRenderer = xyPlot.getRenderer();
    int limit = xySeriesList.xySeriesCollection().getSeriesCount();
    for (int index = 0; index < limit; ++index) {
      xyItemRenderer.setSeriesPaint(index, colorDataIndexed.getColor(index));
      xyItemRenderer.setSeriesStroke(index, stroke);
    }
    xyPlot.setRangeGridlinePaint(Color.LIGHT_GRAY);
    xyPlot.setDomainGridlinePaint(Color.LIGHT_GRAY);
    xyPlot.getDomainAxis().setLowerMargin(0.0);
    xyPlot.getDomainAxis().setUpperMargin(0.0);
    LegendTitle legendTitle = new LegendTitle(xyItemRenderer);
    legendTitle.setPosition(RectangleEdge.TOP);
    jFreeChart.addLegend(legendTitle);
    if (axisClipX != null) {
      NumberAxis numberAxis = (NumberAxis) jFreeChart.getXYPlot().getDomainAxis();
      numberAxis.setRange(axisClipX.min().number().doubleValue(), axisClipY.max().number().doubleValue());
    }
    if (axisClipY != null) {
      NumberAxis numberAxis = (NumberAxis) jFreeChart.getXYPlot().getRangeAxis();
      numberAxis.setRange(axisClipY.min().number().doubleValue(), axisClipY.max().number().doubleValue());
    }
    return jFreeChart;
  }
}
