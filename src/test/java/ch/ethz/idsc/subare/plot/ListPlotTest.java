// code by jph
package ch.ethz.idsc.subare.plot;

import java.io.File;
import java.io.IOException;

import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;

import ch.ethz.idsc.tensor.alg.Sort;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.NormalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;
import ch.ethz.idsc.tensor.pdf.UniformDistribution;
import junit.framework.TestCase;

public class ListPlotTest extends TestCase {
  public void testSimple() throws IOException {
    Distribution uniform = UniformDistribution.unit();
    Distribution normal = NormalDistribution.standard();
    XYSeriesList xySeriesList = new XYSeriesList();
    xySeriesList.addSeries(Sort.of(RandomVariate.of(uniform, 100)), RandomVariate.of(uniform, 100));
    XYSeries xySeries = xySeriesList.addSeries(Sort.of(RandomVariate.of(uniform, 100)), RandomVariate.of(normal, 100));
    xySeries.setKey("normal");
    ListPlot listPlot = new ListPlot(xySeriesList);
    listPlot.setAxisLabelX("domain");
    listPlot.setAxisLabelY("values");
    // listPlot.setPlotLabel("major title");
    JFreeChart jFreeChart = listPlot.jFreeChart();
    File file = HomeDirectory.Pictures(getClass().getSimpleName() + ".png");
    // assertFalse(file.isFile());
    ChartUtils.saveChartAsPNG(file, jFreeChart, 800, 600);
    // assertTrue(file.isFile());
    // file.delete();
    // assertFalse(file.isFile());
  }
}
