// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.io.IOException;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.img.ArrayPlot;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Put;

/* package */ class OptimistsExact extends AbstractExact {
  public OptimistsExact(Supplier<Agent> sup1, Supplier<Agent> sup2, int epochs) {
    super(sup1, sup2, epochs);
    // ---
    contribute(new Integer[] { 0 }, new Integer[] { 0 });
    contribute(new Integer[] { 0, 0 }, new Integer[] { 1 });
    contribute(new Integer[] { 0, 1 }, new Integer[] { 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 0, 0 });
    contribute(new Integer[] { 1 }, new Integer[] { 0, 1 });
    contribute(new Integer[] { 1 }, new Integer[] { 1 });
  }

  public static void showOne() {
    Supplier<Agent> sup1 = //
        () -> new OptimistAgent(2, RationalScalar.of(40, 10), RationalScalar.of(10, 100));
    Supplier<Agent> sup2 = //
        () -> new OptimistAgent(2, RationalScalar.of(38, 10), RationalScalar.of(10, 100));
    OptimistsExact optimistsExact = new OptimistsExact(sup1, sup2, 200);
    System.out.println(optimistsExact.getExpectedRewards());
  }

  public static void main(String[] args) throws IOException {
    Tensor init = Subdivide.of(RationalScalar.of(31, 10), RationalScalar.of(60, 10), 1024 - 1); // 120
    final int n = init.length();
    Tensor res = Array.zeros(n, n);
    Tensor expectedRewards = Array.zeros(n, n);
    IntStream.range(0, n).parallel().forEach( //
        px -> {
          Scalar c0 = init.Get(px);
          IntStream.range(0, n).forEach( //
              py -> {
                Scalar c1 = init.Get(py);
                Supplier<Agent> sup1 = //
                    () -> new OptimistAgent(2, c0, RationalScalar.of(10, 100));
                Supplier<Agent> sup2 = //
                    () -> new OptimistAgent(2, c1, RationalScalar.of(10, 100));
                OptimistsExact exact = new OptimistsExact(sup1, sup2, 50);
                // row.append(Join.of(Tensors.of(c0, c1), exact.getExpectedRewards()));
                expectedRewards.set(exact.getExpectedRewards(), px, py);
              });
        });
    {
      Tensor tensor = Rescale.of(expectedRewards.get(Tensor.ALL, Tensor.ALL, 0));
      Tensor image = ArrayPlot.of(tensor, ColorDataGradients.CLASSIC);
      Export.of(HomeDirectory.Pictures("opts.png"), ImageResize.nearest(image, 1));
    }
    Put.of(HomeDirectory.file("optimist"), res);
  }
}
