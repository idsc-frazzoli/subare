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
  private boolean isLegended = false;

  public Histogram(CategoryDataset categoryDataset) {
    this.categoryDataset = categoryDataset;
    for (int index = 0; index < categoryDataset.getRowCount(); ++index)
      isLegended |= !categoryDataset.getRowKey(index).toString().isEmpty();
  }

  @Override
  public JFreeChart jFreeChart() {
    JFreeChart jFreeChart = ChartFactory.createBarChart( //
        plotLabel, axisLabelX, axisLabelY, categoryDataset, //
        PlotOrientation.VERTICAL, //
        isLegended, // legend
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
    for (int index = 0; index < categoryDataset.getRowCount(); ++index) {
      jFreeChart.getCategoryPlot().getRenderer().setSeriesPaint(index, colorDataIndexed.getColor(index));
      jFreeChart.getCategoryPlot().getRenderer().setSeriesOutlinePaint(index, colorDataIndexed.getColor(index).darker());
    }
    for (int index = 0; index < categoryDataset.getRowCount(); ++index)
      jFreeChart.getCategoryPlot().getRenderer().setSeriesOutlineStroke(index, stroke);
    return jFreeChart;
  }
}
