// code by gjoel, jph
package ch.ethz.idsc.subare.plot;

import java.util.Map;
import java.util.Map.Entry;

import org.jfree.data.category.DefaultCategoryDataset;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Range;

public enum CategoryDatasets {
  ;
  /** @param map
   * @param binSize
   * @return */
  public static DefaultCategoryDataset create(Map<String, Tensor> map, Scalar binSize) {
    DefaultCategoryDataset defaultCategoryDataset = new DefaultCategoryDataset();
    for (Entry<String, Tensor> entry : map.entrySet()) {
      Tensor vector = entry.getValue();
      Tensor tensor = Range.of(0, vector.length() + 1).multiply(binSize);
      for (int j = 0; j < vector.length(); ++j)
        defaultCategoryDataset.addValue(vector.Get(j).number(), entry.getKey(), binName(tensor, j));
    }
    return defaultCategoryDataset;
  }

  private static String binName(Tensor binSize, int i) {
    return "[" + binSize.get(i) + ", " + binSize.get(i + 1) + ")";
  }
}
