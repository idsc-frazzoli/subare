// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.io.IOException;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.img.ArrayPlot;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Put;

/* package */ class UCBsExact extends AbstractExact {
  public UCBsExact(Supplier<Agent> sup1, Supplier<Agent> sup2, int epochs) {
    super(sup1, sup2, epochs);
    // ---
    contribute(new Integer[] { 0 }, new Integer[] { 0 });
    contribute(new Integer[] { 0 }, new Integer[] { 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 0 });
    contribute(new Integer[] { 1 }, new Integer[] { 1 });
  }

  public static void showOne() {
    Supplier<Agent> sup1 = //
        () -> new UCBAgent(2, RationalScalar.of(10, 10));
    Supplier<Agent> sup2 = //
        () -> new UCBAgent(2, RationalScalar.of(8, 10));
    UCBsExact exact = new UCBsExact(sup1, sup2, 200);
    System.out.println(exact.getExpectedRewards());
  }

  public static void main(String[] args) throws IOException {
    Tensor init = Tensors.vector(i -> RationalScalar.of(40 + i, 80), 80); // 80
    Tensor expectedRewards = Array.zeros(init.length(), init.length());
    int px = 0;
    Tensor res = Tensors.empty();
    for (Tensor c0 : init) {
      Tensor row = Tensors.empty();
      int py = 0;
      for (Tensor c1 : init) {
        Supplier<Agent> sup1 = //
            () -> new UCBAgent(2, (Scalar) c0);
        Supplier<Agent> sup2 = //
            () -> new UCBAgent(2, (Scalar) c1);
        UCBsExact exact = new UCBsExact(sup1, sup2, 50);
        row.append(Join.of(Tensors.of(c0, c1), exact.getExpectedRewards()));
        expectedRewards.set(exact.getExpectedRewards(), px, py);
        ++py;
      }
      res.append(row);
      ++px;
    }
    {
      Tensor rescale = Rescale.of(expectedRewards.get(Tensor.ALL, Tensor.ALL, 0));
      Tensor image = ArrayPlot.of(rescale, ColorDataGradients.CLASSIC);
      Export.of(HomeDirectory.Pictures("ucbs.png"), ImageResize.nearest(image, 2));
    }
    Put.of(HomeDirectory.file("ucb"), res);
  }
}
