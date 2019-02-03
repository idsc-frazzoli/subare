// code by gjoel, jph
package ch.ethz.idsc.subare.plot;

import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAnchor;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.category.CategoryDataset;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/Histogram.html">Histogram</a> */
public class Histogram extends AbstractPlotBuilder {
  private final CategoryDataset categoryDataset;

  /** @param bins vector or tensor containing vectors with the respective histogram values
   * @param binSize used beforehand to produce bins
   * @param axisLabelY
   * @param axisLabelX
   * @param labels for the respective histogram columns
   * @return .png image file of the plot
   * @throws Exception */
  public Histogram(CategoryDataset categoryDataset) {
    this.categoryDataset = categoryDataset;
  }

  @Override
  public JFreeChart jFreeChart() {
    JFreeChart jFreeChart = ChartFactory.createBarChart( //
        plotLabel, axisLabelX, axisLabelY, categoryDataset, //
        PlotOrientation.VERTICAL, //
        true, // legend
        false, // tooltips
        false); // urls
    jFreeChart.getCategoryPlot().getDomainAxis().setLowerMargin(0.0);
    jFreeChart.getCategoryPlot().getDomainAxis().setUpperMargin(0.0);
    jFreeChart.getCategoryPlot().getDomainAxis().setCategoryMargin(0.0);
    jFreeChart.getCategoryPlot().getDomainAxis().setCategoryLabelPositions(CategoryLabelPositions.UP_90);
    jFreeChart.getCategoryPlot().setRangeGridlinePaint(Color.LIGHT_GRAY);
    jFreeChart.getCategoryPlot().setRangeGridlinesVisible(true);
    jFreeChart.getCategoryPlot().setDomainGridlinePaint(Color.LIGHT_GRAY);
    jFreeChart.getCategoryPlot().setDomainGridlinesVisible(true);
    jFreeChart.getCategoryPlot().setDomainGridlinePosition(CategoryAnchor.START);
    BarRenderer barRenderer = new BarRenderer();
    barRenderer.setDrawBarOutline(true);
    jFreeChart.getCategoryPlot().setRenderer(barRenderer);
    for (int i = 0; i < categoryDataset.getRowCount(); ++i) {
      jFreeChart.getCategoryPlot().getRenderer().setSeriesPaint(i, colorDataIndexed.getColor(i));
      jFreeChart.getCategoryPlot().getRenderer().setSeriesOutlinePaint(i, colorDataIndexed.getColor(i).darker());
    }
    for (int i = 0; i < categoryDataset.getRowCount(); ++i)
      jFreeChart.getCategoryPlot().getRenderer().setSeriesOutlineStroke(i, stroke);
    return jFreeChart;
  }
}
