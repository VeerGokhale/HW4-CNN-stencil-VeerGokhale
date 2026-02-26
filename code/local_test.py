import torch
import torch.nn as nn
import numpy as np

from convolution.manual_convolution import ManualConv2d


def sample_test():
    """Basic test comparing ManualConv2d to PyTorch's Conv2d."""
    input_data = torch.randn(1, 3, 10, 10)

    out_channels = 16
    kernel_size = 3
    padding = 1

    stu_conv2d = ManualConv2d(in_channels=3, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, use_bias=False)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(3, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Sample test passed!")


def padding_test_same():
    """Test SAME padding (padding=1 for 3x3 kernel maintains size)."""
    input_data = torch.randn(1, 1, 4, 4)

    stu_conv2d = ManualConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, use_bias=False)
    stu_conv2d.filters.data.fill_(1.0)
    stu_output = stu_conv2d(input_data)

    assert stu_output.shape == input_data.shape, f"Expected {input_data.shape}, got {stu_output.shape}"

    true_conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    true_conv2d.weight.data.fill_(1.0)
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Same padding test passed!")


def padding_test_valid():
    """Test VALID padding (padding=0)."""
    input_data = torch.randn(1, 1, 4, 4)

    stu_conv2d = ManualConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, use_bias=False)
    stu_conv2d.filters.data.fill_(1.0)
    stu_output = stu_conv2d(input_data)

    assert stu_output.shape == (1, 1, 2, 2), f"Expected (1, 1, 2, 2), got {stu_output.shape}"

    true_conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
    true_conv2d.weight.data.fill_(1.0)
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Valid padding test passed!")


def base_case_test():
    """Test with 1x1 kernel."""
    input_data = torch.randn(1, 3, 1, 1)

    out_channels = 16
    kernel_size = 1

    stu_conv2d = ManualConv2d(in_channels=3, out_channels=out_channels,
                              kernel_size=kernel_size, padding=0, use_bias=False)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(3, out_channels, kernel_size=kernel_size, padding=0, bias=False)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Base case test passed!")


def weird_shapes_1_same():
    """Test with non-square input (tall and narrow) with SAME-style padding."""
    input_data = torch.randn(4, 2, 100, 3)

    out_channels = 16
    padding = 1

    stu_conv2d = ManualConv2d(in_channels=2, out_channels=out_channels,
                              kernel_size=3, padding=padding, use_bias=False)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(2, out_channels, kernel_size=3, padding=padding, bias=False)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Weird shapes 1 same test passed!")


def weird_shapes_1_valid():
    """Test with non-square input (tall and narrow) with VALID padding."""
    input_data = torch.randn(4, 2, 100, 3)

    out_channels = 16
    kernel_size = 3
    padding = 0

    stu_conv2d = ManualConv2d(in_channels=2, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, use_bias=False)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(2, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Weird shapes 1 valid test passed!")


def weird_shapes_2_same():
    """Test with non-square input (short and wide) with SAME-style padding."""
    input_data = torch.randn(4, 2, 3, 100)

    out_channels = 16
    kernel_size = 3
    padding = 1

    stu_conv2d = ManualConv2d(in_channels=2, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, use_bias=False)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(2, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Weird shapes 2 same test passed!")


def weird_shapes_2_valid():
    """Test with non-square input (short and wide) with VALID padding."""
    input_data = torch.randn(4, 2, 3, 100)

    out_channels = 16
    kernel_size = 2
    padding = 0

    stu_conv2d = ManualConv2d(in_channels=2, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, use_bias=False)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(2, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Weird shapes 2 valid test passed!")


def bias_test():
    """Test that bias is correctly applied."""
    input_data = torch.randn(2, 3, 5, 5)

    out_channels = 8
    kernel_size = 3
    padding = 1

    stu_conv2d = ManualConv2d(in_channels=3, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, use_bias=True)
    stu_output = stu_conv2d(input_data)

    true_conv2d = nn.Conv2d(3, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
    true_conv2d.weight.data = stu_conv2d.filters.data.clone()
    true_conv2d.bias.data = stu_conv2d.bias.data.clone()
    true_output = true_conv2d(input_data)

    np.testing.assert_allclose(stu_output.detach().numpy(), true_output.detach().numpy(), rtol=1e-4, atol=1e-6)
    print("Bias test passed!")


if __name__ == "__main__":
    """
    Uncomment tests to run sanity checks throughout the assignment. These will not be graded
    on gradescope but cover similar edge cases. This way, you can upload to gradescope less frequently.
    """
    ### Simple Tests to Check Conv2D Layers
    sample_test()
    base_case_test()

    ### Tests to Verify Proper Padding Setups
    padding_test_same()
    padding_test_valid()

    ## Tests to Verify Complex Shape Handling
    weird_shapes_1_same()
    weird_shapes_1_valid()
    weird_shapes_2_same()
    weird_shapes_2_valid()

    ## Test bias
    bias_test()