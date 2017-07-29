// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.io.IOException;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.util.Colorscheme;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.opt.Interpolation;

class OptimistsExact extends AbstractExact {
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

  private static final Tensor BASE = Tensors.vector(255);

  public static void main(String[] args) throws IOException {
    Tensor init = Subdivide.of(RationalScalar.of(31, 10), RationalScalar.of(60, 10), 120);
    // System.out.println(init);
    Tensor res = Tensors.empty();
    Tensor expectedRewards = Array.zeros(init.length(), init.length());
    int px = 0;
    for (Tensor c0 : init) {
      Tensor row = Tensors.empty();
      int py = 0;
      for (Tensor c1 : init) {
        Supplier<Agent> sup1 = //
            () -> new OptimistAgent(2, (Scalar) c0, RationalScalar.of(10, 100));
        Supplier<Agent> sup2 = //
            () -> new OptimistAgent(2, (Scalar) c1, RationalScalar.of(10, 100));
        OptimistsExact exact = new OptimistsExact(sup1, sup2, 50);
        row.append(Join.of(Tensors.of(c0, c1), exact.getExpectedRewards()));
        expectedRewards.set(exact.getExpectedRewards(), px, py);
        ++py;
      }
      res.append(row);
      ++px;
    }
    {
      Tensor rescale = Rescale.of(expectedRewards.get(Tensor.ALL, Tensor.ALL, 0));
      Interpolation colorscheme = Colorscheme.CLASSIC;
      rescale = rescale.map(scalar -> colorscheme.get(BASE.multiply(scalar)));
      Export.of(UserHome.Pictures("opts.png"), ImageResize.nearest(rescale, 2));
    }
    Put.of(UserHome.file("optimist"), res);
  }
}
