// code by jph
package ch.ethz.idsc.subare.plot;

import java.awt.BasicStroke;
import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Sort;
import ch.ethz.idsc.tensor.alg.Transpose;
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
    Map<String, Tensor> map = new LinkedHashMap<>();
    map.put("uniform", Transpose.of(Tensors.of(Sort.of(RandomVariate.of(uniform, 100)), RandomVariate.of(uniform, 100))));
    map.put("normal", Transpose.of(Tensors.of(Sort.of(RandomVariate.of(uniform, 100)), RandomVariate.of(normal, 100))));
    XYDataset xyDataset = XYDatasets.create(map);
    ListPlot listPlot = new ListPlot(xyDataset);
    listPlot.setStroke(new BasicStroke(1f));
    JFreeChart jFreeChart = listPlot.jFreeChart();
    File file = HomeDirectory.Pictures(getClass().getSimpleName() + ".png");
    assertFalse(file.isFile());
    ChartUtils.saveChartAsPNG(file, jFreeChart, 800, 600);
    assertTrue(file.isFile());
    file.delete();
    assertFalse(file.isFile());
  }
}
