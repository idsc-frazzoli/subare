// code by jph, fluric
package ch.ethz.idsc.subare.plot;

import java.awt.BasicStroke;
import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.title.LegendTitle;
import org.jfree.chart.ui.RectangleEdge;
import org.jfree.data.xy.XYDataset;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ListPlot.html">ListPlot</a> */
public class ListPlot extends AbstractPlotBuilder {
  private final XYDataset xyDataset;

  public ListPlot(XYDataset xyDataset) {
    this.xyDataset = xyDataset;
    setStroke(new BasicStroke(2f));
  }

  @Override
  public JFreeChart jFreeChart() {
    JFreeChart jFreeChart = ChartFactory.createXYLineChart( //
        plotLabel, axisLabelX, axisLabelY, xyDataset, PlotOrientation.VERTICAL, //
        false, // legend
        false, // tooltips
        false); // urls
    for (int k = 0; k < xyDataset.getSeriesCount(); ++k)
      jFreeChart.getXYPlot().getRenderer().setSeriesPaint(k, colorDataIndexed.getColor(k));
    for (int k = 0; k < xyDataset.getSeriesCount(); ++k)
      jFreeChart.getXYPlot().getRenderer().setSeriesStroke(k, stroke);
    jFreeChart.getXYPlot().setRangeGridlinePaint(Color.LIGHT_GRAY);
    jFreeChart.getXYPlot().setDomainGridlinePaint(Color.LIGHT_GRAY);
    jFreeChart.getXYPlot().getDomainAxis().setLowerMargin(0.0);
    jFreeChart.getXYPlot().getDomainAxis().setUpperMargin(0.0);
    LegendTitle legendTitle = new LegendTitle(jFreeChart.getXYPlot().getRenderer());
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
